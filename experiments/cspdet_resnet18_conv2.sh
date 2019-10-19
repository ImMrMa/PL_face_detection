cd src
# train
CUDA_VISIBLE_DEVICES=2,3 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_conv2 --arch cspfpn --batch_size 32 --lr 5e-4 --lr_step 70,80 --gpus 0,1 --num_workers 3 --dataset wider_csp
