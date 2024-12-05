import os
import argparse
import logging
import math
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path

import numpy as np
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image


from constants import ASPECT_RATIO


from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
# from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
# from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

import decord
import numpy as np
import cv2



import math

import matplotlib

import os


import torch

from mimicmotion.dwpose.wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWposeDetector:
    """
    A pose detect method for image-like data.

    Parameters:
        model_det: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/yolox_l.onnx
        model_pose: (str) serialized ONNX format model path, 
                    such as https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.onnx
        device: (str) 'cpu' or 'cuda:{device_id}'
    """
    def __init__(self, model_det, model_pose, device='cpu'):
        self.pose_estimation = Wholebody(model_det=model_det, model_pose=model_pose, device=device)

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            ###### subset: frames x 18
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1
            # subset: [[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]]


            # un_visible = subset < 0.3
            # candidate[un_visible] = -1
            # foot = candidate[:, 18:24]
            faces = candidate[:, 24:92]
            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

            return pose

dwprocessor = DWposeDetector(
    model_det="models/DWPose/yolox_l.onnx",
    model_pose="models/DWPose/dw-ll_ucoco_384.onnx",
    device=device)


eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

# def draw_bodypose(canvas, candidate, subset, score):
#     H, W, C = canvas.shape
#     candidate = np.array(candidate)
#     subset = np.array(subset)

#     stickwidth = 4

#     limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
#                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
#                [1, 16], [16, 18], [3, 17], [6, 18]]

#     colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
#               [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
#               [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

#     for i in range(17):
#         for n in range(len(subset)):
#             index = subset[n][np.array(limbSeq[i]) - 1]
#             conf = score[n][np.array(limbSeq[i]) - 1]
#             if conf[0] < 0.3 or conf[1] < 0.3:
#                 continue
#             Y = candidate[index.astype(int), 0] * float(W)
#             X = candidate[index.astype(int), 1] * float(H)
#             mX = np.mean(X)
#             mY = np.mean(Y)
#             length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
#             angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
#             polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
#             cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

#     canvas = (canvas * 0.6).astype(np.uint8)

#     for i in range(18):
#         for n in range(len(subset)):
#             index = int(subset[n][i])
#             if index == -1:
#                 continue
#             x, y = candidate[index][0:2]
#             conf = score[n][i]
#             x = int(x * W)
#             y = int(y * H)
#             cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

#     return canvas


def draw_bodypose(canvas, candidate, subset, score):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [1,16]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(15):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if conf[0] < 0.3 or conf[1] < 0.3:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(16):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas




def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2), 
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores):
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])

    ########################################### draw hand pose #####################################################
    canvas = draw_handpose(canvas, hands, pose['hands_score'])

    

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)



def get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        sample_stride: int=1):
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if ref_pose['bodies']['score'].shape[0] > 0 and ref_pose['bodies']['score'][0][i] > 0.3]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    # sample_stride = 10
    detected_poses = [dwprocessor(frm) for frm in vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()]
    detected_bodies = np.stack([p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,ref_keypoint_id]
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 

    total_poses = []
    for i, detected_pose in enumerate(detected_poses):
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        #hhhhhhhhhh: (18, 2) (2, 21, 2)         
        concatenated = np.concatenate([detected_pose['bodies']['candidate'], detected_pose['hands'].reshape(-1,2)], axis=0)
        total_poses.append(concatenated)

        ################# modified
        im = draw_pose(detected_pose, height, width)
        cv2.imwrite('tmp_ske/tmp_ske_'+str(i)+'.png', im.transpose(1,2,0))
        output_pose.append(np.array(im))

    # np.save('total_poses.npy', np.array(total_poses))
    # print('hhhhhhhhhh saved poses')
    # np.save('original_poses.npy', np.stack(output_pose))
    return np.stack(output_pose)



def save_video_from_frames(frames, output_path, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()


def get_image_pose(ref_image):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)


logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(video_path, image_path, resolution=576, sample_stride=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): input video pose path
        image_path (str): reference image path
        resolution (int, optional):  Defaults to 576.
        sample_stride (int, optional): Defaults to 2.
    """
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels) # (c, h, w)
    h, w = image_pixels.shape[-2:]
    ############################ compute target h/w according to original aspect ratio ###############################
    if h>w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()
    ##################################### get image&video pose value #################################################
    #image_pixels shape: 1024, 576, 3
    image_pose = get_image_pose(image_pixels)
    #image_pose: 3, 1024, 576
    video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
    #video pose shape: 530, 3, 1024, 576
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    
    image_pixels = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    #image_pixels shape: 1, 3, 1024, 576
    return torch.from_numpy(pose_pixels.copy()) / 127.5 - 1, torch.from_numpy(image_pixels) / 127.5 - 1


def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    image_pixels = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    pose_pixels = pose_pixels.unsqueeze(0).to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels, image_pose=pose_pixels, num_frames=pose_pixels.size(1),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    for vid_idx in range(video_frames.shape[0]):
        # deprecated first frame because of ref image
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames




import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel
from mimicmotion.modules.pose_net import PoseNet
from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae").half()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])



# def create_pipeline(infer_config, device):
#     mimicmotion_models = MimicMotionModel(infer_config.base_model_path).to(device=device).eval()
#     mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location=device), strict=False)
#     pipeline = MimicMotionPipeline(
#         vae=mimicmotion_models.vae, 
#         image_encoder=mimicmotion_models.image_encoder, 
#         unet=mimicmotion_models.unet, 
#         scheduler=mimicmotion_models.noise_scheduler,
#         feature_extractor=mimicmotion_models.feature_extractor, 
#         pose_net=mimicmotion_models.pose_net
#     )
#     return pipeline

def create_pipeline(infer_config, device):
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path).to(device=device).eval()
    # mimicmotion_models.load_state_dict(torch.load(infer_config.ckpt_path, map_location=device), strict=False)


    pose_net = mimicmotion_models.pose_net
    pose_net_state_dict = torch.load("../SVD_Xtend-main/tmp_all_data/humanvid_3point/pose_guider-50000.pth", map_location=device)
    # pose_net_state_dict = torch.load("../SVD_Xtend-main/TrainSvd_fullhands_slim_alldata_8version_3pointface/pose_guider-150000.pth", map_location=device)
    # pose_net_state_dict = torch.load("../SVD_Xtend-main/TrainSvd_fullhands_slim_alldata_8version_3pointface/pose_guider-68000.pth", map_location=device)
    # pose_net_state_dict = torch.load("../SVD_Xtend-main/tmp_all_data/humanvid_3point/pose_guider-50000.pth", map_location=device)

    pose_net.load_state_dict(pose_net_state_dict, strict=True)

    unet = mimicmotion_models.unet
    unet.load_state_dict(torch.load("../SVD_Xtend-main/tmp_all_data/humanvid_3point/denoising_unet-50000.pth"))
    # unet.load_state_dict(torch.load("../SVD_Xtend-main/TrainSvd_fullhands_slim_alldata_8version_3pointface/denoising_unet-150000.pth"))
    # unet.load_state_dict(torch.load("../SVD_Xtend-main/TrainSvd_fullhands_slim_alldata_8version_3pointface/denoising_unet-68000.pth"))
    # unet.load_state_dict(torch.load("../SVD_Xtend-main/tmp_all_data/humanvid_3point/denoising_unet-50000.pth"))

    



    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae, 
        image_encoder=mimicmotion_models.image_encoder, 
        unet=unet, 
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=pose_net
    )
    return pipeline





@torch.no_grad()
def main(args):
    if not args.no_use_float16 :
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(infer_config, device)

    for task in infer_config.test_case:
        ############################################## Pre-process data ##############################################
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, task.ref_image_path, 
            resolution=task.resolution, sample_stride=task.sample_stride
        )
        ########################################### Run MimicMotion pipeline ###########################################
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        ################################### save results to output folder. ###########################################
        save_to_mp4(
            _video_frames, 
            # f"{args.output_dir}/{os.path.basename(task.ref_video_path).split('.')[0]}" \
            # f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4",
            #################### modified
            f"benchmark_videos_final_tiktok_total/{os.path.basename(task.ref_video_path).split('.')[0]}_{os.path.basename(task.ref_image_path).split('.')[0]}.mp4",
            fps=task.fps,
        )

def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml") #ToDo
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    set_logger(args.log_file \
               if args.log_file is not None else f"{args.output_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    main(args)
    logger.info(f"--- Finished ---")

# CUDA_VISIBLE_DEVICES=3 python inference_3pointsface.py --inference_config configs/test.yaml
# CUDA_VISIBLE_DEVICES=3 python inference_3pointsface.py --inference_config configs/test3_point_face.yaml
# CUDA_VISIBLE_DEVICES=3 python inference_3pointsface.py --inference_config configs/test3_point_face_tiktok.yaml
