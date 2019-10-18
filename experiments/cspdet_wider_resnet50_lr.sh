cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim sgd --exp_id wider_resnet50_follow_csp_lr_plateau --arch cspfpn --batch_size 14 --lr 2e-4 --lr_step 60 --gpus 0,1 --num_workers 4 --dataset wider_csp
