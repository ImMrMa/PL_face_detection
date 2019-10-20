cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_conv3 --arch cspfpn --batch_size 32 --lr 5e-4 --lr_step 70,80 --gpus 0,1,2,3 --num_workers 3 --dataset wider_csp
