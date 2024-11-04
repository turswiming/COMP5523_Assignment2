import cv2
import numpy as np
from utils import imread, imshow, write_and_show, destroyAllWindows

def keypoint_match(img1, img2, max_n_match=100, draw=True):
    # make sure they are of dtype uint8
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: convert to grayscale by `cv2.cvtColor`
    img1_gray = ...
    img2_gray = ...


    # TODO: detect keypoints and generate descriptor by `sift.detectAndCompute`
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = ...
    keypoints2, descriptors2 = ...


    # draw keypoints
    if draw:
        # TODO: draw keypoints on image1 and image2 by `cv2.drawKeypoints`
        pass


    # TODO: Knn match and Lowe's ratio test
    matcher = cv2.FlannBasedMatcher_create()
    ...


    # TODO: select best `max_n_match` matches
    ...


    return keypoints1, keypoints2, match


def draw_match(img1, keypoints1, img2, keypoints2, match, savename):
    img1 = np.uint8(img1)
    img2 = np.uint8(img2)

    # TODO: draw matches by `cv2.drawMatches`
    match_draw = ...


    write_and_show(savename, match_draw)


def transform(img, img_kps, dst_kps, H, W):
    '''
    Transfrom img such `img_kps` are aligned with `dst_kps`.
    H: height of output image
    W: width of output image
    '''
    # TODO: get transform matrix by `cv2.findHomography`
    T, status = ...


    # TODO: apply transform by `cv2.warpPerspective`
    transformed = ...


    return transformed


if __name__ == '__main__':
    ## read images
    img1 = imread('image/left2.jpg')
    img2 = imread('image/right2.jpg')

    ## find keypoints and matches
    keypoints1, keypoints2, match = keypoint_match(img1, img2, max_n_match=1000)

    draw_match(img1, keypoints1, img2, keypoints2,
               match, savename='results/1.4_match.jpg')

    # get all matched keypoints
    keypoints1 = np.array([keypoints1[m.queryIdx].pt for m in match])
    keypoints2 = np.array([keypoints2[m.trainIdx].pt for m in match])

    ## Align img2 to img1
    H, W = img1.shape[:2]
    W = W*2
    new_img2 = transform(img2, keypoints2, keypoints1, H, W)
    write_and_show('results/1.6_transformed.jpg', new_img2)

    # resize img1
    new_img1 = np.hstack([img1, np.zeros_like(img1)])
    # write_and_show('results/new_img1.jpg', new_img2)

    # TODO: average `new_img1` and `new_img2`
    cnt = ...

    stack = ...

    write_and_show('results/1.7_stack.jpg', stack)

    destroyAllWindows()
