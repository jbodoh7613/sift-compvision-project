# John Bodoh Computer Vision Assignment 4
# Implement SIFT feature extraction algorithm

import numpy as np
import cv2 as cv

"""
Returns scale space for a given image as a tuple of tuples, where each tuple within the main tuple contains the images of a single octave.
Each octave consists of multiple "scales" of the given image, where each scale is a Gaussian blur of the image which uses a sigma value k times the sigma value used for the previous scale
For each successive octave, the image that is used is the image used in the previous octave, resized to half of its previous size
"""
# first starting sigma can be 1.6 or half of 2^(1/2)
# starting sigma for each new octave should be middle sigma value used for previous octave
def create_scale_space(img: np.ndarray, num_octaves: int = 4, num_scales: int = 5, sigma: float = 1.6, k: float = 1.414214):
    middle_scale = num_scales//2 # We use integer division as the floor of the true quotient will be the value of the looping variable when the middle scale is encountered
    octave_list = []
    octave_list_tuples = []
    for i in range(num_octaves):
        octave_list.append([])
    img_scaled = img.copy()
    current_sigma = sigma
    for i in range(num_octaves):
        for j in range(num_scales):
            octave_list[i].append(cv.GaussianBlur(img_scaled,(5,5),current_sigma))
            if(j == middle_scale):
                next_sigma = current_sigma
            current_sigma = k*current_sigma
        current_sigma = next_sigma
        img_scaled = cv.resize(img_scaled, (int(img_scaled.shape[0]/1.414), int(img_scaled.shape[1]/1.414)))
    for i in range(len(octave_list)):
        octave_list_tuples.append(tuple(octave_list[i]))
    return(tuple(octave_list_tuples))

def create_scale_space_test():
    blocks = cv.imread('blocks_L-150x150.png', cv.IMREAD_GRAYSCALE)
    scale_space = create_scale_space(blocks)
    for i in range(len(scale_space)):
        for j in range(len(scale_space[i])):
            cv.imshow('(%d, %d)'.format(i, j), scale_space[i][j])
            cv.waitKey(-1)

def main():
    create_scale_space_test()

if __name__ == '__main__':
    main()