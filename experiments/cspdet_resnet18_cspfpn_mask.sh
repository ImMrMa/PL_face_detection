cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam  --exp_id wider_resnet18_csp_mask --mask --arch cspfpn --batch_size 20 --lr 5e-4 --lr_step 70,80  --gpus 0,1 --num_workers 4 --dataset wider_csp 
