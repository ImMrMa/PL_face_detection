cd src
# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py cspdet --optim adam --exp_id wider_resnet18_csp_adam_s1_conv4_conv2_gn_pretrained  --arch cspfpn --batch_size 48 --lr 1e-3 --lr_step 70,90 --gpus 0,1,2,3  --num_workers 4 --dataset wider_csp
