cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam --exp_id wider_resnet18_ori_conv3 --num_epochs 180 --arch cspfpn --batch_size 32 --lr 1e-3 --lr_step 70,80 --gpus 0,1 --num_workers 3 --dataset wider_csp
