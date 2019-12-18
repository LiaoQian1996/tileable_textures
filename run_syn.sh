CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir result \
    --target_dir ./textures/texture_${i}.png \
    --texture_shape 4096 4096 \
    --top_style_layer VGG54 \
    --max_iter 50 \
    --pyrm_layers 6 \
    --W_tv 0.001 \
    --pad 32 \
    #--print_loss \
    #--vgg_ckpt ./vgg19/
sleep 1
python patches2img.py --path texture_${i}_