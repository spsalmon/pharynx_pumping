from worm_tools import straightening, masks, midline
import os
from skimage.filters import threshold_yen
from skimage.util import img_as_ubyte
from tifffile import imread, imwrite
import cv2
import numpy as np
from joblib import Parallel, delayed

input_dir = "/mnt/external.data/TowbinLab/kstojanovski/20220401_Ti2_20x_160-182-190_pumping_25C_20220401_173300_429/movies/"
output_dir = "/mnt/external.data/TowbinLab/kstojanovski/20220401_Ti2_20x_160-182-190_pumping_25C_20220401_173300_429/analysis/str_pharynx_videos/"

def get_biggest_object(mask):
	# Get the mask's connected components
	_,labels = cv2.connectedComponents(mask)
	# Find the biggest object in the frame
	label_counts = np.bincount(labels.flatten())

	if len(label_counts) >= 2:
		biggest_object_label = np.argsort(label_counts)[-2]
	else:
		biggest_object_label = 0

	biggest_object_mask = img_as_ubyte(labels == biggest_object_label)
	return biggest_object_mask

def get_pharynx_mask(pharynx_vid):
    pharynx_masks = np.zeros_like(pharynx_vid)
    for i, pharynx_frame in enumerate(pharynx_vid):
        # Threshold the pharynx frame to obtain a binary mask of the pharynx
        pharynx_threshold = 0.75*threshold_yen(pharynx_frame)
        pharynx_mask = img_as_ubyte(pharynx_frame > pharynx_threshold)
        # Apply median blur to the pharynx mask
        pharynx_mask = cv2.medianBlur(pharynx_mask, 11)
        pharynx_mask = get_biggest_object(pharynx_mask)
        pharynx_masks[i] = pharynx_mask
    return pharynx_masks

def get_images_of_point(images, point):
    """
    Given a list of images and a point, return a list of image names that contain the point in their file names.
    
    Parameters:
        images (list): List of image file names.
        point (str): Point to search for in image file names.
    
    Returns:
        list: List of image names that contain the point in their file names.
    """
    
    # Initialize empty list to store image names
    image_names = []
    
    # Iterate through list of images
    for image in images:
        # Check if point is in the image file name
        if point in os.path.basename(image):
            # If point is found, append image name to list
            image_names.append(image)
    
    # Return list of image names
    return image_names

def straighten_pharynx_vid(video_path):
    print(f'Straightening {video_path}')
    pharynx_video = imread(video_path)
    try:
        mask = get_pharynx_mask(pharynx_video)
        straightener = straightening.Warper.from_img(pharynx_video, mask)
        straightened_video = straightener.warp_3D_img(pharynx_video)
        imwrite(os.path.join(output_dir, os.path.basename(video_path)), straightened_video)
    except:
        print(f'Could not straighten {video_path}')
        with open('/home/spsalmon/straightening/errors.txt', 'a') as file:
            file.write(f'{video_path}\n')


    # mask = get_pharynx_mask(pharynx_video)
    # straightener = straightening.Warper.from_img(pharynx_video, mask)
    # straightened_video = straightener.warp_3D_img(pharynx_video)
    # imwrite(os.path.join(output_dir, os.path.basename(video_path)), straightened_video)

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    # Get list of all pharynx videos
    pharynx_videos = sorted([os.path.join(input_dir, video) for video in os.listdir(input_dir)])
    # Straighten pharynx videos
    Parallel(n_jobs=-1)(delayed(straighten_pharynx_vid)(video) for video in pharynx_videos)
