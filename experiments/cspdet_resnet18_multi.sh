cd src
# train
CUDA_VISIBLE_DEVICES=0,1 python main.py cspdet --optim adam --multi_scale --exp_id wider_resnet18_csp_multi_conf  --arch cspfpnmulti --batch_size 40 --lr 5e-4 --lr_step 70,80 --resume --gpus 0,1 --num_workers 3 --dataset wider_csp 
