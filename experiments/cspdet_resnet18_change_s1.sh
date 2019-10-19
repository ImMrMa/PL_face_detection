cd src
# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_change_s1 --arch cspfpn --batch_size 32 --lr 5e-4 --lr_step 70,80 --gpus 0,1,2,3 --num_workers 3 --dataset wider_csp
