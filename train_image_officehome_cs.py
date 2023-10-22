from __future__ import print_function
import argparse
import torch.optim as optim
from utils import *
from models.basenet import *
from torchvision import transforms, datasets
import torch.nn.functional as F
import os
import time
import numpy as np
import warnings
#from data_loader.folder import ImageFolder_ind
from data_list import DAImageList, ForeverDataIterator
import CSutils
import data_loader_home
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser(description='ImageClef Classification')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N', help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.0003)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP', help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K', help='how many steps to repeat the generator update')
parser.add_argument('--num_layer', type=int, default=2, metavar='K', help='how many layers for classifier')
parser.add_argument('--train_path', type=str, default='dataset/clef/i', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default='dataset/clef/p', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--class_num', type=int, default='31', metavar='B', help='The number of classes')
parser.add_argument('--gmn_N', type=int, default='12', metavar='B', help='The number of classes to calulate gradient similarity')
parser.add_argument('--resnet', type=str, default='50', metavar='B', help='which resnet 18,50,101,152,200')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--iter_per_epoch', type=int, default=200)
# source='image_list/dslr.txt'
# target='image_list/amazon.txt'

parser.add_argument('--source', type=str, default='Art')
parser.add_argument('--target', type=str, default='Clipart')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.cuda.set_device(args.gpu)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)



train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
lr = args.lr

def scheduler(optimizer_, init_lr_, decay_step_, gamma_):
    class DecayLRAfter:
        def __init__(self, optimizer, init_lr, decay_step, gamma):
            self.init_lr = init_lr
            self.gamma = gamma
            self.optimizer = optimizer
            self.iter_num = 0
            self.decay_step = decay_step

        def get_lr(self) -> float:
            if ((self.iter_num + 1) % self.decay_step) == 0:
                lr = self.init_lr * self.gamma
                self.init_lr = lr

            return self.init_lr

        def step(self):
            """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
            lr = self.get_lr()
            for param_group in self.optimizer.param_groups:
                if 'lr_mult' not in param_group:
                    param_group['lr_mult'] = 1.
                param_group['lr'] = lr * param_group['lr_mult']

            self.iter_num += 1

        def __str__(self):
            return str(self.__dict__)

    return DecayLRAfter(optimizer_, init_lr_, decay_step_, gamma_)

def build_data_loaders(root_src, source, target, workers, batch_size, num_classes=31):
    ImageNetNormalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    class ResizeImage(object):
        """Resize the input PIL Image to the given size.

        Args:
            size (sequence or int): Desired output size. If size is a sequence like
                (h, w), output size will be matched to this. If size is an int,
                output size will be (size, size)
        """

        def __init__(self, size):
            if isinstance(size, int):
                self.size = (int(size), int(size))
            else:
                self.size = size

        def __call__(self, img):
            th, tw = self.size
            return img.resize((th, tw))

    train_transform = transforms.Compose([
        ResizeImage(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ImageNetNormalize
    ])

    # Test Transform....
    val_transform = transforms.Compose([
        ResizeImage(256),  
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ImageNetNormalize
    ])

    cls_name = [f"{i}" for i in range(num_classes)]
    train_source = DataLoader(DAImageList(root_src, cls_name, source, transform=train_transform),
                              batch_size=batch_size,
                              shuffle=True, num_workers=workers, drop_last=True, pin_memory=True
                              )
    train_target = DataLoader(DAImageList(root_src, cls_name, target, transform=train_transform),
                              batch_size=batch_size,
                              shuffle=True, num_workers=workers, drop_last=True, pin_memory=True
                              )

    # validation target data....
    val_loader = DataLoader(DAImageList(root_src, [i for i in range(num_classes)], target, transform=val_transform),
                            batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=True)

    return train_source, train_target, val_loader


option = 'resnet' + args.resnet
G = ResBottle(option)
F1 = ResClassifier(num_classes=65, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F2 = ResClassifier(num_classes=65, num_layer=num_layer, num_unit=G.output_num(), middle=1000)
F1.apply(weights_init)
F2.apply(weights_init)

start = -1

if args.cuda:
    G.cuda()
    F1.cuda()
    F2.cuda()

#wd=0.002
wd=0.0005
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD([{"params": G.features.parameters(), "lr_mult": 0.1},
                            {"params": G.bottleneck.parameters(), "lr_mult": 1.0}], 
                            lr=args.lr, weight_decay=wd)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                            weight_decay=wd)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam([{"params": G.features.parameters(), "lr_mult": 0.1},
                            {"params": G.bottleneck.parameters(), "lr_mult": 1.0}], 
                            lr=args.lr, weight_decay=wd)
    
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)





def sample_batch(train_source, train_target, device):
    x_s, labels_s = next(train_source)
    x_t, _ = next(train_target)
    x_s = x_s.to(device)
    x_t = x_t.to(device)
    labels_s = labels_s.to(device)
    return x_s, x_t, labels_s


#root_src='/data/Codes/Domain-Adaption/dataset/OfficeHome'
source=args.source
target=args.target 
workers = 4
iter_per_epoch = args.iter_per_epoch #200 #2000
print("source: {}, target: {}".format(source, target))
print(args)

opt_g_schedule = scheduler(optimizer_g, args.lr, decay_step_=iter_per_epoch * 5, gamma_=0.5)
opt_f_schedule = scheduler(optimizer_f, args.lr, decay_step_=iter_per_epoch * 5, gamma_=0.5)

# train_source, train_target, test_loader = build_data_loaders(root_src=root_src,
#                                                             source=os.path.join(root_src, source),
#                                                             target=os.path.join(root_src, target), workers=workers,
#                                                             batch_size=batch_size)

dataset = 'OfficeHome'
data_folder = '/data/Codes/Domain-Adaption/dataset/OfficeHome'

train_source = data_loader_home.loader(dataset,data_folder,source,
                                        batch_size,alexnet=False,train=True)

train_target = data_loader_home.loader(dataset,data_folder,target,
                                        batch_size,alexnet=False,train=True)
# Testing loader
test_loader = data_loader_home.loader(dataset,data_folder,target,
                                        batch_size,alexnet=False,train=False)

print(len(train_source))
len_source = len(train_source)
# train_target = ForeverDataIterator(train_target)
# train_source = ForeverDataIterator(train_source)



def train(num_epoch):
    
    criterion = nn.CrossEntropyLoss()
    criterion_w = Weighted_CrossEntropy
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #CrossEntropyLabelSmooth(num_classes = 12, epsilon=0.1).da()
    for ep in range(num_epoch):
        
        since = time.time()

        print("Obtaining target label...")
        # mem_label = obtain_label(data_loader_T_no_shuffle, G, F1, F2, args)
        # mem_label = torch.from_numpy(mem_label).cuda()
        # for batch_idx, data in enumerate(dataset):
        #     if dataset.stop_S:
        #         brea
        
        #for batch_idx in range(iter_per_epoch):
        for batch_idx, ((data_s, label_s), (data_t, label_t)) in enumerate(zip(train_source,train_target)):
            opt_g_schedule.step()
            opt_f_schedule.step()
            
            if data_s.shape[0] != 32:
                print(data_s.shape, data_t.shape, len(train_source), len(train_target))
            # batch data loading...
            #data_s, data_t, label_s = sample_batch(train_source, train_target, device)

            G.train()
            F1.train()
            F2.train()

            if args.cuda:
                data_s, label_s = data_s.cuda(), label_s.cuda()
                #data_t, label_t = data_t.cuda(), label_t.cuda()
                data_t = data_t.cuda()
                #if ep > start:
                #    pseudo_label_t = pseudo_label_t.cuda()
            data_all = Variable(torch.cat((data_s, data_t), 0))
            label_s = Variable(label_s)
            bs = len(label_s)

            """source domain discriminative"""
            # Step A train all networks to minimize loss on source
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)


            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)

            Z_s = output[:bs, :]
            Z_t = output[bs:, :]
            Z_s, Z_t = F.normalize(Z_s, p=2, dim=1), F.normalize(Z_t, p=2, dim=1)

            CS_loss = CSutils.CS(Z_s, Z_t, 1, median_sigma=0) 
            output_t1, output_t2 = F.normalize(output_t1, p=2, dim=1), F.normalize(output_t2, p=2, dim=1)
            output_s1, output_s2 = F.normalize(output_s1, p=2, dim=1), F.normalize(output_s2, p=2, dim=1)
            
            CCS_loss = CSutils.CondCSD(Z_s, Z_t, output_s1, output_t1, 
                                     sigma = 1,
                                     median_sigma=0) 

            CCS_loss += CSutils.CondCSD(Z_s, Z_t, output_s2, output_t2, 
                                     sigma = 1,
                                     median_sigma=0) 
            
            
            all_loss = loss1 + loss2 + 0.1 * entropy_loss + CS_loss + CCS_loss #+ 0.1 * supervision_loss
            all_loss.backward()
            optimizer_g.step()
            optimizer_f.step()

            """target domain diversity"""
            # Step B train classifier to maximize CDD loss
            optimizer_g.zero_grad()
            optimizer_f.zero_grad()
            output = G(data_all)
            output1 = F1(output)
            output2 = F2(output)
            output_s1 = output1[:bs, :]
            output_s2 = output2[:bs, :]
            output_t1 = output1[bs:, :]
            output_t2 = output2[bs:, :]
            output_t1_s = F.softmax(output_t1)
            output_t2_s = F.softmax(output_t2)

            loss1 = criterion(output_s1, label_s)
            loss2 = criterion(output_s2, label_s)

            entropy_loss = Entropy(output_t1_s)
            entropy_loss += Entropy(output_t2_s)
            # entropy_loss = - torch.mean(torch.log(torch.mean(output_t1_s, 0) + 1e-6))
            # entropy_loss -= torch.mean(torch.log(torch.mean(output_t1_s, 0) + 1e-6))

            #loss_dis = discrepancy(output_t1, output_t2)
            Z_t = output[bs:, :]
            Z_t = F.normalize(Z_t, p=2, dim=1)
            output_t1, output_t2 = F.normalize(output_t1, p=2, dim=1), F.normalize(output_t2, p=2, dim=1)
            CCS_loss = CSutils.CondCSD(Z_t, Z_t, output_t1, output_t2, 
                                     sigma = 1,
                                     median_sigma=0) 
            loss_dis = CCS_loss
            all_loss = loss1 + loss2 - 1.0 * loss_dis + 0.1 * entropy_loss
            all_loss.backward()
            optimizer_f.step()

            """target domain discriminability"""
            # Step C train genrator to minimize CDD loss
            for i in range(num_k):
                optimizer_g.zero_grad()
                optimizer_f.zero_grad()

                output = G(data_all)
                output1 = F1(output)
                output2 = F2(output)
                output_s1 = output1[:bs, :]
                output_s2 = output2[:bs, :]
                output_t1 = output1[bs:, :]
                output_t2 = output2[bs:, :]
                output_t1_s = F.softmax(output_t1)
                output_t2_s = F.softmax(output_t2)

                entropy_loss = Entropy(output_t1_s)
                entropy_loss += Entropy(output_t2_s)


                Z_s = output[:bs, :]
                Z_t = output[bs:, :]
                Z_s, Z_t = F.normalize(Z_s, p=2, dim=1), F.normalize(Z_t, p=2, dim=1)

                CS_loss = CSutils.CS(Z_s, Z_t, 1, median_sigma=0) 

                output_t1, output_t2 = F.normalize(output_t1, p=2, dim=1), F.normalize(output_t2, p=2, dim=1)
                CCS_loss = CSutils.CondCSD(Z_t, Z_t, output_t1, output_t2, 
                                        sigma = 1,
                                        median_sigma=0) 
                loss_dis = CCS_loss + CS_loss
            

                all_loss = 1.0 * loss_dis + 0.1 * entropy_loss 

                all_loss.backward()
                optimizer_g.step()

            if batch_idx % args.log_interval == 0:
                print(
                    'Train Ep: {} [{}/{} ({:.6f}%)]\tLoss1: {:.6f}\tLoss2: {:.6f}\t CDD: {:.6f} Entropy: {:.6f} '.format(
                        ep, batch_idx, len(train_source), 100. * batch_idx / len(train_source),
                        loss1.item(), loss2.item(), loss_dis.item(), entropy_loss.item()))


        # test
        test(ep)
        print('time:', time.time() - since)
        print('-' * 100)

def test(epoch):
    G.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct_add = 0
    size = 0
    print('-' * 100, '\nTesting')
    dataset_test = test_loader
    for batch_idx, data in enumerate(dataset_test):
        # if dataset_test.stop_T:
        #     break
        if args.cuda:
            img, label = data
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img, volatile=True), Variable(label)
        with torch.no_grad():
            output = G(img)
            output1 = F1(output)
            output2 = F2(output)
        test_loss += F.nll_loss(output1, label).item()
        output_add = output1 + output2
        pred = output_add.data.max(1)[1]
        correct_add += pred.eq(label.data).cpu().sum()
        size += label.data.size()[0]


    test_loss /= len(test_loader)  # loss function already averages over batch size
    print('Epoch: {:d} Test set: Average loss: {:.6f}, Accuracy: {}/{} ({:.6f}%)'.format(
        epoch, test_loss, correct_add, size, 100. * float(correct_add) / size))



train(args.epochs + 1)
