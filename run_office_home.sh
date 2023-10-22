# ['Art', 'Clipart', 'Product', 'Real_World']
# CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Clipart  --target Real_World   > log_officehome_c_r.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Product  --target Clipart   > log_officehome_p_c.log 2>&1 &
# wait
#CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Product  --target Real_World   > log_officehome_p_r.log 2>&1 &
#CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Real_World  --target Art   > log_officehome_r_a.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Real_World  --target Art   > log_officehome_r_a.log 2>&1 &
# wait
# CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Real_World  --target Art   > log_officehome_r_a.log 2>&1 &


CUDA_VISIBLE_DEVICES=0  nohup python train_image_officehome_cs.py  --lr 0.004 --source Art  --target Clipart   > log_officehome_a_c.log 2>&1 &
