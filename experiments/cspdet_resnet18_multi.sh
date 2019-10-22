cd src
# train
CUDA_VISIBLE_DEVICES=3 python main.py cspdet --optim adam --multi_scale --exp_id wider_resnet18_csp_multi --arch cspfpnmulti --batch_size 10 --lr 5e-4 --lr_step 70,80 --gpus 0 --num_workers 3 --dataset wider_csp
