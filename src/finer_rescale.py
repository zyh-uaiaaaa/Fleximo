import cv2
import math
import numpy as np
from tqdm import tqdm

ref_pose = np.load('ref_pose.npy')
# tgt_pose = np.load('tgt_pose.npy')
ref_pose[:,0] *= 576
ref_pose[:,1] *= 1024
# tgt_pose[:,0] *= 576
# tgt_pose[:,1] *= 1024





def draw_bodypose(canvas, candidate):
    H, W, C = canvas.shape
    candidate = np.array(candidate)

    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],  [1, 15], [15, 17], [1,16], [16, 18]]



    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],   [85, 0, 255], [170, 0, 255],[255, 0, 255], \
               [255, 0, 170], [255,   0,  85]]


    # print('hhhhhhh ', candidate)
    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        # if candidate[index.astype(int)[0], 0]< 0.1 or candidate[index.astype(int)[0], 1]<0.1 or candidate[index.astype(int)[1], 0]<0.1 or candidate[index.astype(int)[1], 1]<0.1:
        #     continue
        ################# modified #################
        Y = candidate[index.astype(int), 0] * float(W/576)
        X = candidate[index.astype(int), 1] * float(H/1024)
        # Y = candidate[index.astype(int), 0]
        # X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        x, y = candidate[i][0:2]
        x = int(x * W/576)
        y = int(y * H/1024)
        # x, y = int(x), int(y)
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    return canvas

def draw_bodypose_wrong(canvas, candidate):
    H, W, C = canvas.shape
    candidate = np.array(candidate)

    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1,16], [16, 18], [1, 15], [15, 17]]



    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],   [85, 0, 255], [170, 0, 255],[255, 0, 255], \
               [255, 0, 170], [255,   0,  85]]


    # print('hhhhhhh ', candidate)
    for i in range(17):
        index = np.array(limbSeq[i]) - 1
        # if candidate[index.astype(int)[0], 0]< 0.1 or candidate[index.astype(int)[0], 1]<0.1 or candidate[index.astype(int)[1], 0]<0.1 or candidate[index.astype(int)[1], 1]<0.1:
        #     continue
        ################# modified #################
        Y = candidate[index.astype(int), 0] * float(W/576)
        X = candidate[index.astype(int), 1] * float(H/1024)
        # Y = candidate[index.astype(int), 0]
        # X = candidate[index.astype(int), 1]
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, colors[i])

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        x, y = candidate[i][0:2]
        x = int(x * W/576)
        y = int(y * H/1024)
        # x, y = int(x), int(y)
        cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    return canvas








def draw_pose(pose, H, W, ref_w=2160):
    candidate = pose

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    # sr = 1

    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    #candidate.shape: 18x2, subset.shape:18, score:18

    canvas = draw_bodypose(canvas, candidate)
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB)
def draw_pose_wrong(pose, H, W, ref_w=2160):
    candidate = pose

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    # sr = 1

    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    #candidate.shape: 18x2, subset.shape:18, score:18

    canvas = draw_bodypose_wrong(canvas, candidate)
    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB)













video_writer = cv2.VideoWriter('body_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (576,1024))

total_body = np.load('body.npy')
for i in tqdm(range(total_body.shape[0])):
    tgt_pose = total_body[i]

    tgt_pose[:,0] *= 576
    tgt_pose[:,1] *= 1024



    # 1-2 shoulder
    d_ref_21 = np.linalg.norm(ref_pose[2] - ref_pose[1])
    d_tgt_21 = np.linalg.norm(tgt_pose[2] - tgt_pose[1])
    direction_21 = (tgt_pose[2] - tgt_pose[1]) / d_tgt_21
    tgt_new_2 = tgt_pose[1] + direction_21 * d_ref_21
    translation = tgt_new_2 - tgt_pose[2]
    tgt_pose[4] = tgt_pose[4] + translation
    tgt_pose[3] = tgt_pose[3] + translation
    tgt_pose[2] = tgt_new_2


    # # 1-5 shoulder
    d_ref_51 = np.linalg.norm(ref_pose[5] - ref_pose[1])
    d_tgt_51 = np.linalg.norm(tgt_pose[5] - tgt_pose[1])
    direction_51 = (tgt_pose[5] - tgt_pose[1]) / d_tgt_51
    tgt_new_5 = tgt_pose[1] + direction_51 * d_ref_51
    translation = tgt_new_5 - tgt_pose[5]
    tgt_pose[7] = tgt_pose[7] + translation
    tgt_pose[6] = tgt_pose[6] + translation
    tgt_pose[5] = tgt_new_5



    ##### neck
    d_ref = np.linalg.norm(ref_pose[0] - ref_pose[1])
    d_tgt = np.linalg.norm(tgt_pose[0] - tgt_pose[1])
    direction = (tgt_pose[0] - tgt_pose[1]) / d_tgt
    tgt_new_0 = tgt_pose[1] + direction * d_ref
    translation = tgt_new_0 - tgt_pose[0]
    tgt_pose[14] = tgt_pose[14] + translation
    tgt_pose[15] = tgt_pose[15] + translation
    tgt_pose[16] = tgt_pose[16] + translation
    tgt_pose[17] = tgt_pose[17] + translation
    tgt_pose[0] = tgt_new_0

    ######14-16 head
    # d_ref_1517 = np.linalg.norm(ref_pose[17] - ref_pose[15])
    # d_tgt_1416 = np.linalg.norm(tgt_pose[16] - tgt_pose[14])
    # direction_1416 = (tgt_pose[16] - tgt_pose[14]) / d_tgt_1416
    # tgt_pose[16] = tgt_pose[14] + direction_1416 * d_ref_1517


    # d_ref = np.linalg.norm(ref_pose[15] - ref_pose[0])
    # d_tgt = np.linalg.norm(tgt_pose[14] - tgt_pose[0])
    # direction = (tgt_pose[14] - tgt_pose[0]) / d_tgt
    # tgt_new_14 = tgt_pose[0] + direction * d_ref
    # translation = tgt_new_14 - tgt_pose[14]
    # tgt_pose[16] = tgt_pose[16] + translation
    # tgt_pose[14] = tgt_new_14

    # ######15-17 head
    # d_ref_1416 = np.linalg.norm(ref_pose[16] - ref_pose[14])
    # d_tgt_1517 = np.linalg.norm(tgt_pose[17] - tgt_pose[15])
    # direction_1517 = (tgt_pose[17] - tgt_pose[15]) / d_tgt_1517
    # tgt_pose[17] = tgt_pose[15] + direction_1517 * d_ref_1416


    # d_ref = np.linalg.norm(ref_pose[14] - ref_pose[0])
    # d_tgt = np.linalg.norm(tgt_pose[15] - tgt_pose[0])
    # direction = (tgt_pose[15] - tgt_pose[0]) / d_tgt
    # tgt_new_15 = tgt_pose[0] + direction * d_ref
    # translation = tgt_new_15 - tgt_pose[15]
    # tgt_pose[17] = tgt_pose[17] + translation
    # tgt_pose[15] = tgt_new_15












    im = draw_pose_wrong(tgt_pose, 1024, 576)
    im_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    video_writer.write(im_bgr)






video_writer.release()

# print(np.linalg.norm(ref_pose[3] - ref_pose[2]), np.linalg.norm(tgt_pose[3] - tgt_pose[2]))



# im = draw_pose(tgt_pose, 1024, 576)
# im_bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
# cv2.imwrite('tgt_pose.jpg', im_bgr)


