cd src
# train
CUDA_VISIBLE_DEVICES=2,3 python main.py cspdet --optim adam --multi_scale --exp_id wider_resnet50_csp_multi  --arch cspfpnmulti --resume --batch_size 18 --lr 1e-3 --lr_step 70,80  --gpus 0,1 --resume --num_workers 4 --dataset wider_csp 
