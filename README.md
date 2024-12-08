# Fleximo | Work In Progress
### The GitHub repository is still being updated. Stay tuned!


The official implementation of the paper [Fleximo: Towards Flexible Text-to-Human Motion Video Generation](https://arxiv.org/abs/2411.19459)

## Motion-to-Video Generation
The training code is primarily adapted from the SVD fine-tuning code available at [pixeli99/SVD_Xtend](https://github.com/pixeli99/SVD_Xtend/blob/main/train_svd.py). We sincerely thank pixeli99 for providing the fine-tuning framework.

Our video generation model is fine-tuned using the pre-trained weights from [MimicMotion](https://github.com/Tencent/MimicMotion). We extend our gratitude to the creators of MimicMotion for making their pre-trained model available.

To train an image-to-video generation model conditioned on pose sequences, you will need to download the pre-trained weights of both MimicMotion and SVD.


The running command is:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch train_video_generation.py \
    --pretrained_model_name_or_path="/stabilityai/stable-video-diffusion-img2vid-xt-1-1" \
    --per_gpu_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=100000 \
    --width=576 \
    --height=1024 \
    --checkpointing_steps=5000 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="fp16" \
    --enable_xformers_memory_efficient_attention \
    --validation_steps=0 \
    --gradient_checkpointing
```


The training data we used is available at [HumanVid](https://github.com/zhenzhiwang/HumanVid). We extend our gratitude to the authors for sharing their dataset. After fine-tuning the model, you will obtain a video generation model capable of producing high-quality human motion videos based on an input image and a pose sequence.


## Text-to-Motion Generation
To enable text condition instead of pose sequence condtion, we incoporate a pre-trained text-to-motion module [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) for motion generation. Our work is inspired by [hmtv](https://github.com/CSJasper/HMTV), from which you can acquire the code to project text-generated 3D mesh into 2D skeleton. 

## Skeleton Adapter 
It is non-trivial to enable text-conditioned motion video generation. To compensate the missing information of T2M-GPT generated skeletons, we train a skeleton adapter using the train_skeleton_adapter.py.

The skeleton adapter can take into handless skeleton videos and generate suitable hand movements. It is initialized from the SVD weights and using pose videos with and without hands to train. The pose videos both with and without hands are extracted from human motion videos.


##  Anchor-Point Based Rescale
We also propose an anchor-point based rescale and a refinement process to solve the misalignment between the T2M-GPT generated skeletons and the skeletons extracted by DWPose like in MimicMotion.  The anchor-point based rescale is implemented in finer_scale.py and the refinment process is implemented in the inference.py.

##  Inference
The inference process of our model is similar to MimicMotion. To run inference, run

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py --inference_config configs/test.yaml
```
configs/test.yaml is similar to MimicMotion


##  Some Results
### condition image
![demo11](https://github.com/user-attachments/assets/be565953-e45c-4ff5-bd84-1d03c1207f3c)

### a man is running

https://github.com/user-attachments/assets/6074cc4c-fef1-4ec5-9bda-e6dbc4666d6d

### a man is waving his hand

https://github.com/user-attachments/assets/ec74bd87-c484-40b7-91a5-ee123c2fab36

### a man is crossing his arms


https://github.com/user-attachments/assets/7cdd3020-e3f4-4389-ad91-8637114eec48

### a man is golfing



https://github.com/user-attachments/assets/4bc15210-3e61-4d15-bbda-76f15bcc6005

