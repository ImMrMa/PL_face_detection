cd src
# train
CUDA_VISIBLE_DEVICES=4,5,6,7 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_s1_all_gn  --arch cspfpn --batch_size 36 --lr 1e-3 --lr_step 70,90 --gpus 0,1,2,3  --num_workers 4 --dataset wider_csp
