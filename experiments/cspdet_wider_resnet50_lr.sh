cd src
# train
CUDA_VISIBLE_DEVICES=1,2 python main.py cspdet --optim sgd --exp_id wider_resnet50_follow_csp_lr_plateau --arch cspfpn --batch_size 30 --lr 1e-1 --lr_step 60 --gpus 0,1 --num_workers 10 --resume --dataset wider_csp
