# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import sys

##################################
# Network & Variable
##################################
def weights_init(m):
    """Initialize network parameters."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.05)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)         
        
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)
    
def to_var(x):
    """Convert numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to numpy."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()   

def classification_accuracy(data_loader,DNN,FC):
    with torch.no_grad():
        correct = 0
        for batch_idx, (X, lab) in enumerate(data_loader):
            X, lab = to_var(X), to_var(lab).long().squeeze()
            _, prob = FC(DNN(X))
            plab = prob.data.max(1)[1]
            correct += plab.eq(lab.data).cpu().sum()
        accuracy = correct.item() / len(data_loader.dataset)
        return accuracy
    
    
def inv_lr_scheduler(param_lr, optimizer, iter_num, gamma, power, init_lr=0.001, weight_decay=0.0005):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (1 + gamma * iter_num) ** (-power)
    i = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_lr[i]
        param_group['weight_decay'] = weight_decay * 2
        i += 1
    return optimizer


schedule_dict = {"inv":inv_lr_scheduler}

##################################
# Objective Functions
##################################
# Cross-Entropy Loss
NLL_loss = torch.nn.NLLLoss().cuda() 
def Cross_Entropy(prob,lab):
    CE_loss = NLL_loss(torch.log(prob+1e-4), lab)
    return CE_loss

# Entropy Loss
def Entropy(prob):
    num_sam = prob.shape[0]
    Entropy = -(prob.mul(prob.log()+1e-4)).sum()
    return Entropy/num_sam



def compute_sigma(H):
    dists = torch.pdist(H)
    sigma = dists.median()/2
    return sigma.detach()

def GaussianMatrix(X,Y,sigma, if_use_cdist=False, median_sigma = False):

    if not if_use_cdist:
        size1 = X.size()
        size2 = Y.size()
        G = (X*X).sum(-1)
        H = (Y*Y).sum(-1)
        Q = G.unsqueeze(-1).repeat(1,size2[0])
        R = H.unsqueeze(-1).T.repeat(size1[0],1)
        #print(G.shape, R.shape, X.shape, Y.shape, Q.shape, R.shape)
        H = Q + R - 2*X@(Y.T)
    else:
        H = torch.cdist(X, Y, p=2)**2

    if sigma > 0:
        H = torch.exp(-H/2/sigma**2)
    else:
        if median_sigma:
            sigma = compute_sigma(H)
            H = torch.exp(-H/2/sigma / 2)
        else:
            sigma = H.mean().detach()
            H = torch.exp(-H/sigma)
    return H



# CKB loss
def CondCSD(x1,x2,y1,y2,sigma = -1, if_use_cdist=False, median_sigma=False): # conditional cs divergence
    # Input: N x d

    # x1 = torch.tensor(x1)
    # x2 = torch.tensor(x2)
    # y1 = torch.tensor(y1)
    # y2 = torch.tensor(y2)
    
    K1 = GaussianMatrix(x1,x1,sigma, if_use_cdist, median_sigma) # a lot of 0 (1560)

    K2 = GaussianMatrix(x2,x2,sigma, if_use_cdist, median_sigma) # 1560 0
    
    L1 = GaussianMatrix(y1,y1,sigma, if_use_cdist, median_sigma) 
    L2 = GaussianMatrix(y2,y2,sigma, if_use_cdist, median_sigma)

    #print(x1.shape, x2.shape, y1.shape, y2.shape, K1.shape, K2.shape, L1.shape, L2.shape)    
    
    K12 = GaussianMatrix(x1,x2,sigma, if_use_cdist, median_sigma) # nan happens  1600 0 ---> all zeros --> makes the later part nan 
    L12 = GaussianMatrix(y1,y2,sigma, if_use_cdist, median_sigma) # 
    
    K21 = GaussianMatrix(x2,x1,sigma, if_use_cdist, median_sigma) # nan happens  1600 0
    L21 = GaussianMatrix(y2,y1,sigma, if_use_cdist, median_sigma)

    H1 = K1*L1 # 1560 0
    self_term1 = (H1.sum(-1)/((K1.sum(-1))**2)).sum(0) # 
    
    H2 = K2*L2
    self_term2 = (H2.sum(-1)/((K2.sum(-1))**2)).sum(0)
    
    ##################################DEBUG#################################################
    H3 = K12*L12
    cross_term1 = (H3.sum(-1)/((K1.sum(-1))*(K12.sum(-1)))).sum(0) # # nan first happens
    ##################################DEBUG################################################
    H4 = K21*L21
    cross_term2 = (H4.sum(-1)/((K2.sum(-1))*(K21.sum(-1)))).sum(0)
    
    cs1 = -2*torch.log(cross_term1) + torch.log(self_term1) + torch.log(self_term2)
    cs2 = -2*torch.log(cross_term2) + torch.log(self_term1) + torch.log(self_term2)
    
    
    return ((cs1+cs2)/2)

def CS(x1,x2,sigma = 10, if_use_cdist=False, median_sigma=False): # conditional cs divergence
    #x1 = torch.tensor(x1)
    #x2 = torch.tensor(x2)
    
    K1 = GaussianMatrix(x1,x1,sigma, if_use_cdist, median_sigma)
    K2 = GaussianMatrix(x2,x2,sigma, if_use_cdist, median_sigma)
    
    K12 = GaussianMatrix(x1,x2,sigma, if_use_cdist, median_sigma)

    dim1 = K1.shape[0]
    self_term1 = K1.sum()/(dim1**2)
    
    dim2 = K2.shape[0]
    self_term2 = K2.sum()/(dim2**2)
    
    cross_term = K12.sum()/(dim1*dim2)
    
    cs =  -2*torch.log(cross_term) + torch.log(self_term1) + torch.log(self_term2)
   
    return cs


