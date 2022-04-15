# John Bodoh Computer Vision Assignment 4
# Implement SIFT feature extraction algorithm

import numpy as np
import cv2 as cv

"""
Returns scale space for a given image as a tuple of tuples, where each tuple within the main tuple contains the images of a single octave.
Each octave consists of multiple "scales" of the given image, where each scale is a Gaussian blur of the image which uses a sigma value k times the sigma value used for the previous scale
For each successive octave, the image that is used is the image used in the previous octave, resized to half of its previous size.
"""
# first starting sigma can be 1.6 or half of 2^(1/2)
# starting sigma for each new octave should be middle sigma value used for previous octave
def create_scale_space(img: np.ndarray, num_octaves: int = 4, num_scales: int = 5, sigma: float = 1.6, k: float = 1.414214):
    middle_scale = num_scales//2 # We use integer division as the floor of the true quotient will be the value of the looping variable when the middle scale is encountered
    octave_list = [[] for _ in range(num_octaves)]
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
    return(space_list_to_tuple(octave_list))

"""
This function takes a scale space tuple and returns a LoG (Laplacian of Gaussian) space represented as a tuple of tuples, where each tuple within the main tuple consists of the LoG images of its respective octave.
LoG is calculated using Difference of Gaussians, where, for each image in each octave, a LoG image is produced by subtracting the image below from it.
"""
def create_log_space(scale_space: tuple[tuple[np.ndarray, ...], ...]):
    octave_list = [[] for _ in range(len(scale_space))]
    for i in range(len(scale_space)):
        for j in range(len(scale_space[i]) - 1):
            octave_list[i].append(np.subtract(scale_space[i][j+1], scale_space[i][j]))
    return(space_list_to_tuple(octave_list))

# Converts a list of lists of images, representing an image space, to a tuple of tuples of images
def space_list_to_tuple(space_list: list[list[np.ndarray]]):
    return(tuple(tuple(octave) for octave in space_list))

# Loads image as grayscale
def load_image(img: str):
    return(cv.imread(img, cv.IMREAD_GRAYSCALE))

# Display all images in given space
def display_space_images(space: tuple[tuple[np.ndarray, ...], ...]):
    for i in range(len(space)):
        for j in range(len(space[i])):
            cv.imshow('%d, %d'.format(i, j), space[i][j])
            cv.waitKey(-1)

def create_scale_space_test():
    blocks = load_image("blocks_L-150x150.png")
    scale_space = create_scale_space(blocks)
    display_space_images(scale_space)

def create_log_space_test():
    blocks = load_image("blocks_L-150x150.png")
    scale_space = create_scale_space(blocks)
    log_space = create_log_space(scale_space)
    display_space_images(log_space)

def main():
    create_log_space_test()

if __name__ == '__main__':
    main()