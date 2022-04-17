# John Bodoh Computer Vision Assignment 4
# Implement SIFT feature extraction algorithm

import numpy as np
import cv2 as cv

"""
Returns scale space for a given image as a tuple of tuples, where each tuple within the main tuple contains the images of a single octave.
Each octave consists of multiple "scales" of the given image, where each scale is a Gaussian blur of the image which uses a sigma value k times the sigma value used for the previous scale.
For each successive octave, the image that is used is the image used in the previous octave, resized to half of its previous size.
"""
# first starting sigma can be 1.6 or half of 2^(1/2)
# starting sigma for each new octave should be middle sigma value used for previous octave
def create_scale_space(img: np.ndarray, num_octaves: int = 4, num_scales: int = 5, sigma: float = 1.6, k: float = 1.414214):
    middle_scale = num_scales//2 # We use integer division as the floor of the true quotient will be the value of the looping variable when the middle scale is encountered
    octave_list = [[] for _ in range(num_octaves)]
    img_scaled = img.copy()
    current_sigma = sigma
    for octave_index in range(num_octaves):
        for scale_index in range(num_scales):
            octave_list[octave_index].append(cv.GaussianBlur(img_scaled,(5,5),current_sigma))
            if(scale_index == middle_scale):
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
    for octave_index in range(len(scale_space)):
        for scale_index in range(len(scale_space[octave_index]) - 1): # One less DoG image per octave than scaled image
            octave_list[octave_index].append(np.subtract(scale_space[octave_index][scale_index+1], scale_space[octave_index][scale_index]))
    return(space_list_to_tuple(octave_list))

"""
This function takes a log space tuple and checks, for every pixel excluding the ones in the top and bottom scales, whether it has the largest or smallest pixel value among its 26 neighbors.
All pixels that are larger or smaller than their neighbors by more than the given threshold are recorded in the dictionary returned by the function.
The dictionary has key values consisting of tuples containing the x and y coordinates of the point, its scale number, and its octave number respectively, and it has values corresponding to the difference in values the pixel has to its closest neighbor.
Positive values indicate that the given point is a maximum among its neighbors, while negative values indicate that the given point is a minimum among its neighbors.
"""
def create_min_max_dict(log_space: tuple[tuple[np.ndarray, ...], ...], threshold: int = 0):
    min_max_dict = {}
    # 3D sliding window
    for octave_index in range(len(log_space)):
        for scale_index in range(1, len(log_space[octave_index]) - 1): # Exclude top and bottom DoG images
            for global_x in range(1, log_space[octave_index][scale_index].shape[0] - 1): # Do not iterate through border pixels
                for global_y in range(1, log_space[octave_index][scale_index].shape[1] - 1): # Do not iterate through border pixels
                    current_pixel_value = log_space[octave_index][scale_index].item((global_x, global_y))
                    # check neighbors within 3D sliding window
                    larger_value_found = False
                    largest_neighbor_value = 0
                    smaller_equal_value_found = False
                    smallest_neighbor_value = 255
                    not_min_max = False
                    for window_x_offset in range(-1, 2):
                        for window_y_offset in range(-1, 2):
                            for window_scale_offset in range(-1, 2):
                                window_pixel_value = log_space[octave_index][scale_index + window_scale_offset].item((global_x + window_x_offset, global_y + window_y_offset))
                                if(window_pixel_value > current_pixel_value):
                                    larger_value_found = True
                                elif(window_pixel_value <= current_pixel_value):
                                    smaller_equal_value_found = True
                                if(larger_value_found and smaller_equal_value_found):
                                    not_min_max = True
                                    break
                                if(not window_x_offset and not window_y_offset and not window_scale_offset): # Do not include current pixel when calculating largest values of neighbors
                                    break
                                if(window_pixel_value > largest_neighbor_value):
                                    largest_neighbor_value = window_pixel_value
                                if(window_pixel_value < smallest_neighbor_value):
                                    smallest_neighbor_value = window_pixel_value
                            if(not_min_max):
                                break
                        if(not_min_max):
                            break
                    if(not_min_max):
                        break
                    elif((not larger_value_found) and (current_pixel_value - largest_neighbor_value > threshold)):
                        min_max_dict.update({(global_x, global_y, scale_index, octave_index): current_pixel_value - largest_neighbor_value})
                    elif((not smaller_equal_value_found) and (smallest_neighbor_value - current_pixel_value > threshold)):
                        min_max_dict.update({(global_x, global_y, scale_index, octave_index): current_pixel_value - smallest_neighbor_value})
    return min_max_dict

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

def create_min_max_dict_test():
    blocks = load_image("blocks_L-150x150.png")
    scale_space = create_scale_space(blocks)
    log_space = create_log_space(scale_space)
    min_max_dict = create_min_max_dict(log_space)
    print(min_max_dict)

def main():
    create_min_max_dict_test()

if __name__ == '__main__':
    main()