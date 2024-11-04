import cv2
import numpy as np
from tqdm import tqdm
from sift import keypoint_match, draw_match, transform
from utils import read_video_frames, write_frames_to_video, write_and_show, destroyAllWindows, imshow

video_name = 'image/winter_day.mov'
images, fps = read_video_frames(video_name)
n_image = len(images)

# TODO: init panorama
# 2.1
h, w = images[0].shape[:2]
H, W = h, w*3
panorama = np.zeros([H,W,3])
h_start = 0
w_start = W-w
panorama[h_start:h_start+h, w_start:w_start+w, :] = images[0]

trans_sum = np.zeros([H,W,3])
cnt = np.ones([H,W,1])*1e-10
panorama_list = []
for img in tqdm(images[::4], 'processing'):
    # TODO: stitch img to panorama one by one
    # 2.2 align and average
    keypoints1, keypoints2, match = keypoint_match(panorama, img, max_n_match=100,draw=False)
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])

    aligned = transform(img, keypoints2, keypoints1, H, W)


    trans_sum += aligned
    cnt += (aligned != 0).any(2, keepdims=True)
    panorama = trans_sum/cnt
    panorama_list.append(panorama)
    del aligned
    del keypoints1
    del keypoints2
    del match
    #call gc
    import gc
    gc.collect()

    # show
    # imshow('results/2_panorama.jpg', panorama)

write_frames_to_video('results/2.3_panorama_list.mp4', panorama_list, fps/2)

# panorama = algined.mean(0)
write_and_show('results/2.4_panorama.jpg', panorama)

destroyAllWindows()
