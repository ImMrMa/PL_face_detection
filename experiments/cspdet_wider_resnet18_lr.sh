cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam --arch cspfpn --batch_size 32 --lr 2e-4 --lr_step 60,80 --gpus 0,1 --num_workers 4  --dataset wider_csp
