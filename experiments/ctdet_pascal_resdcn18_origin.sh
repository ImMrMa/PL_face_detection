cd src
# train
CUDA_VISIBLE_DEVICES=1 python main.py ctdet --exp_id pascal_resnet101_original_hm_ma --arch resdcn_18 --multi_res --dataset pascal --input_res 512 --batch_size 32 --num_epochs 70  --lr_step 30,45,55 --gpus 1
