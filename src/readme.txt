The training code is highly based on the SVD finetuning code: https://github.com/pixeli99/SVD_Xtend/blob/main/train_svd.py, thanks for pixeli99 for sharing the finetuning code.
Our video generation model is finetuned from the weights of MimicMotion: https://github.com/Tencent/MimicMotion, thanks for their pre-trained model.
To train a image-to-video generation model conditioned on pose sequences, you need to download the pre-trained weights of MimicMotion and SVD.

The running command is : 
CUDA_VISIBLE_DEVICES=0 accelerate launch  train_video_generation.py --pretrained_model_name_or_path="/stabilityai/stable-video-diffusion-img2vid-xt-1-1"    --per_gpu_batch_size=1 --gradient_accumulation_steps=1     --max_train_steps=100000     --width=576    --height=1024     --checkpointing_steps=5000 --checkpoints_total_limit=1     --learning_rate=1e-5 --lr_warmup_steps=0     --seed=123     --mixed_precision="fp16" --enable_xformers_memory_efficient_attention --validation_steps=0 --gradient_checkpointing

The training data we use is downloading from https://github.com/zhenzhiwang/HumanVid. Thanks for the authors sharing their data. After finetuning the model, you can get a video generation model which can generate high-quality human motion videos based on an image and a pose sequence.

