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
def create_scale_space(img: np.ndarray, num_octaves: int = 4, num_scales: int = 5, sigma: float = 1.6, k: float = 1.414214) -> tuple[tuple[np.ndarray, ...], ...]:
    middle_scale = num_scales//2 # We use integer division as the floor of the true quotient will be the value of the looping variable when the middle scale is encountered
    octave_list = [[] for _ in range(num_octaves)]
    img_scaled = img.copy()
    current_sigma = sigma
    for octave_index in range(num_octaves):
        for scale_index in range(num_scales):
            octave_list[octave_index].append(cv.GaussianBlur(img_scaled,(0,0),current_sigma))
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
def create_log_space(scale_space: tuple[tuple[np.ndarray, ...], ...]) -> tuple[tuple[np.ndarray, ...], ...]:
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
def create_min_max_dict(log_space: tuple[tuple[np.ndarray, ...], ...], threshold: int = 0) -> dict[tuple[int, int, int, int], int]:
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


"""
This function takes a min_max_dict as well as a scale space tuple, along with values of sigma and k, and outputs a tuple of all keypoint objects, performing orientation assignment.
"""
def create_keypoints(min_max_dict: dict[tuple[int, int, int, int], int], scale_space: tuple[tuple[np.ndarray, ...], ...], sigma: float = 1.6, k: float = 1.414214) -> tuple[cv.KeyPoint, ...]:
    keypoint_coord_list = list(min_max_dict)
    num_octaves = len(scale_space)
    octave_starting_index_list = []
    octave_starting_index_list.append(sigma)
    keypoint_object_list = []
    for octave_index in range(1, num_octaves):
        octave_starting_index_list.append((num_octaves//2)*k*octave_starting_index_list[octave_index - 1])
    for point in range(len(keypoint_coord_list)):
        point_x_coord = keypoint_coord_list[point][0]
        point_y_coord = keypoint_coord_list[point][1]
        point_scale = keypoint_coord_list[point][2]
        point_octave = keypoint_coord_list[point][3]
        scale_sigma = (point_scale + 1)*octave_starting_index_list[point]
        window_sigma = 1.5*scale_sigma
        window_side_length = int(6.66*window_sigma - 2.22) # The cv2.getGaussianKernel() function, for a given side length value ksize, calculates a sigma value using the equation 0.3*((ksize-1)*0.5 - 1) + 0.8. Solving for ksize, we can calculate a side length of a Gaussian kernel from a given sigma using 6.66*sigma - 2.22
        if(window_side_length % 2 == 0): # window_size_length must be odd
            window_side_length = window_side_length + 1
        window_radius = window_side_length//2
        if(window_radius + 1 > point_x_coord or window_radius + 1 > point_y_coord or window_radius + 1 > len(scale_space[point_octave][point_scale]) - point_x_coord or window_radius + 1 > len(scale_space[point_octave][point_scale]) - point_y_coord): # Do not make keypoint if given point is too close to edge to be centered in an appropriately-sized window
            break
        angle_bin_dict = {k: 0.0 for k in range(36)}
        for window_x_offset in range(-window_radius, window_radius + 1):
            for window_y_offset in range(-window_radius, window_radius + 1):
                x_derivative = scale_space[point_octave][point_scale].item((point_x_coord + window_x_offset + 1, point_y_coord + window_y_offset)) - scale_space[point_octave][point_scale].item((point_x_coord + window_x_offset - 1, point_y_coord + window_y_offset))
                y_derivative = scale_space[point_octave][point_scale].item((point_x_coord + window_x_offset, point_y_coord + window_y_offset + 1)) - scale_space[point_octave][point_scale].item((point_x_coord + window_x_offset, point_y_coord + window_y_offset - 1))
                magnitude = ((x_derivative**2 + y_derivative**2)**0.5)*magnitude_gaussian_weight(window_x_offset, window_y_offset, window_sigma)
                direction = int(np.arctan(float(y_derivative/x_derivative)))
                angle_bin_dict[direction//10] = angle_bin_dict[direction//10] + magnitude
        primary_vector_magnitude = 0
        secondary_vector_direction_list = []
        secondary_vector_magnitude_list = []
        for bin_index in range(36):
            if(angle_bin_dict[bin_index] > primary_vector_magnitude):
                primary_vector_bin = bin_index
        primary_vector_direction = primary_vector_bin + 5
        primary_vector_magnitude = angle_bin_dict[primary_vector_bin]
        for bin_index in range(36):
            if(angle_bin_dict[bin_index] >= primary_vector_magnitude*0.8):
                secondary_vector_direction_list.append(bin_index + 5)
                secondary_vector_magnitude_list.append(angle_bin_dict[bin_index])
        keypoint_object_list.append(cv.KeyPoint(point_x_coord, point_y_coord, window_side_length, primary_vector_direction, 0, point_octave))
        for vector_index in range(len(secondary_vector_direction_list)):
            keypoint_object_list.append(cv.KeyPoint(point_x_coord, point_y_coord, window_side_length, secondary_vector_direction_list[vector_index], 0, point_octave))
    return tuple(keypoint_object_list)

def magnitude_gaussian_weight(x: int, y: int, sigma: float) -> float:
    return(float(1/(2*3.141593))*np.exp(float(-0.5*(x*x + y*y)))/sigma**2)

# Converts a list of lists of images, representing an image space, to a tuple of tuples of images
def space_list_to_tuple(space_list: list[list[np.ndarray]]) -> tuple[tuple[np.ndarray, ...], ...]:
    return(tuple(tuple(octave) for octave in space_list))

# Loads image as grayscale
def load_image(imgpath: str) -> np.ndarray:
    return(cv.imread(imgpath, cv.IMREAD_GRAYSCALE))

# Display all images in given space
def display_space_images(space: tuple[tuple[np.ndarray, ...], ...]):
    for i in range(len(space)):
        for j in range(len(space[i])):
            cv.imshow('%d, %d'.format(i, j), space[i][j])
            cv.waitKey(-1)

def create_scale_space_test(imgpath: np.ndarray):
    img = load_image(imgpath)
    scale_space = create_scale_space(img)
    display_space_images(scale_space)

def create_log_space_test(imgpath: np.ndarray):
    img = load_image(imgpath)
    scale_space = create_scale_space(img)
    log_space = create_log_space(scale_space)
    display_space_images(log_space)

def create_min_max_dict_test(imgpath: np.ndarray):
    img = load_image(imgpath)
    scale_space = create_scale_space(img)
    log_space = create_log_space(scale_space)
    min_max_dict = create_min_max_dict(log_space)
    print(min_max_dict)

# Check and compare output of built-in SIFT function
def sift_test(imgpath: np.ndarray):
    imggray = load_image(imgpath)
    img = cv.imread(imgpath, cv.IMREAD_COLOR)
    sift = cv.SIFT_create()
    kp = sift.detect(imggray, None)
    cv.imshow('SIFT Test', cv.drawKeypoints(imggray, kp, img))
    cv.waitKey(-1)

def main():
    imggray = load_image('blocks_L-150x150.png')
    img = cv.imread('blocks_L-150x150.png', cv.IMREAD_COLOR)
    scale_space = create_scale_space(imggray)
    log_space = create_log_space(scale_space)
    min_max_dict = create_min_max_dict(log_space)
    kp = create_keypoints(min_max_dict, scale_space)
    cv.imshow('SIFT Keypoints', cv.drawKeypoints(imggray, kp, img))
    cv.waitKey(-1)

if __name__ == '__main__':
    main()