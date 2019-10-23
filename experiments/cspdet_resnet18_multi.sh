cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam --multi_scale --exp_id wider_resnet18_csp_multi_s3  --arch cspfpnmulti --batch_size 8 --lr 2e-4 --lr_step 70,80 --gpus 0,1 --num_workers 3 --dataset wider_csp
