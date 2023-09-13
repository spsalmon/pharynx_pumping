import numpy as np
import matplotlib.pyplot as plt
import os
from nd2reader import ND2Reader
import tifffile
import cv2
from time import time
import matplotlib
from skimage.feature import canny
from skimage.util import img_as_ubyte, img_as_uint
from joblib import Parallel, delayed
import logging
from itertools import repeat
logging.basicConfig(level=logging.NOTSET)
from skimage.filters import threshold_otsu

# folder_path = r"/home/spsalmon/pharynx_test/videos/"
folder_path = r"/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/raw/"
output_dir = r"/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/analysis/optimized_cropped_videos/"


os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'cropped_pharynx_videos'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'cropped_body_videos'), exist_ok=True)



def get_bbox(mask, expand=40):
    try:
        # Find the non-zero pixels in the mask
        y_coords, x_coords = np.nonzero(mask)

        # Calculate the minimum and maximum x and y coordinates
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)

        # Expand the bounding box by the specified amount
        y_min -= expand
        x_min -= expand
        height = y_max - y_min + 2 * expand
        width = x_max - x_min + 2 * expand

        return [[y_min, x_min], width, height]
    except ValueError:
        return [[1000, 1000], 10, 10]
	
def segmentation_otsu(image):
	blur = cv2.GaussianBlur(image,(5,5),0)	
	_,mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	disk_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk_kernel)
	return img_as_ubyte(mask)


def get_bbox_of_frame(inputs):
	clahe = cv2.createCLAHE(clipLimit=25, tileGridSize=(8,8))
	images, frame_index = inputs

	# Get the pharynx and body frame
	pharynx_frame = images[frame_index, 0, :, :]
	body_frame = images[frame_index, 1, :, :]

	# Normalize the pharynx and body frame
	max_value = np.iinfo(pharynx_frame.dtype).max
	pharynx_frame_norm = cv2.normalize(pharynx_frame, None, 0, max_value,cv2.NORM_MINMAX)
	body_frame_norm = cv2.normalize(body_frame, None, 0, max_value,cv2.NORM_MINMAX)
 
	pharynx_frame = clahe.apply(pharynx_frame_norm)
	body_frame = clahe.apply(body_frame_norm)

	# Segment the pharynx and body frame
	pharynx_mask = segmentation_otsu(pharynx_frame)
	body_mask = segmentation_otsu(body_frame)

	# Get the body mask's connected components (label each worm with a different number)
	_, body_labels = cv2.connectedComponents(body_mask)
	# Find the biggest worm in the frame
	body_label_counts = np.bincount(body_labels.flatten())

	if len(body_label_counts) >= 2:
		biggest_worm_label = np.argsort(body_label_counts)[-2]
	else:
		biggest_worm_label = 0
	
	# Get the mask of the biggest worm
	biggest_worm_mask = (body_labels == biggest_worm_label)

	# Get the pharynx mask's connected components
	retval_pharynx, pharynx_labels = cv2.connectedComponents(pharynx_mask)

	# Iterate over all the labels to find the pharynx with the greatest overlap with the biggest worm's mask
	overlap_of_labels = []
	for label in range(1, retval_pharynx):
		overlap_count = np.sum(np.logical_and(biggest_worm_mask, pharynx_labels==label))
		overlap_of_labels.append((label, overlap_count))
	
	best_pharynx_label = max(overlap_of_labels, key=lambda x: x[1])[0]
	
	pharynx_mask = (pharynx_labels == best_pharynx_label)

	# Get the bounding box for the pharynx
	pharynx_bbox = get_bbox(pharynx_mask)
	return pharynx_bbox


def resize_bbox(inputs):
	bbox, max_height, max_width = inputs

	height_diff = max_height - bbox[2]
	width_diff = max_width - bbox[1]

	origin = bbox[0]

	origin[0] = max(origin[0] - height_diff // 2, 0)
	origin[1] = max(origin[1] - width_diff // 2, 0)

	return [origin, max_width, max_height]


already_saw_files = []


def crop_pharynx(file):
	if file in already_saw_files:
		return
	extension = os.path.splitext(file)[1]
	if extension != ".nd2":
		return

	logging.info(f"Working on {file}")
	crop_time = time()

	with ND2Reader(file) as images:
		metadata = images.metadata
		nb_frames = metadata['num_frames']
		height = metadata['height']
		width = metadata['width']
		frames = metadata['frames']  # consider only the first 5 frames

		video_numpy = np.empty((nb_frames, 2, height, width), dtype=np.uint16)
		for frame in frames:
			# logging.info(frame)
			video_numpy[frame, 0, :, :] = images.get_frame_2D(c=0, t=frame)
			video_numpy[frame, 1, :, :] = images.get_frame_2D(c=1, t=frame)

		images.close()

	print(f'time for opening the video : {time() - crop_time}')
	start_time = time()
	boxes = Parallel(n_jobs=-1, prefer="threads")(delayed(get_bbox_of_frame)(inp)
												  for inp in zip(repeat(video_numpy), frames))
	print(f'time for getting the bboxs: {time() - start_time}')
	# Initialize the max height and width to 0
	max_height = max_width = 0

	# Iterate over the boxes to find the max height and width
	for bbox in boxes:
		width = bbox[1]
		height = bbox[2]
		max_width = max(max_width, width)
		max_height = max(max_height, height)

	# Add 150 pixels to the max height and width
	max_height += 150
	max_width += 150

	boxes = Parallel(n_jobs=-1, prefer="threads")(delayed(resize_bbox)(inp)
												  for inp in zip(boxes, repeat(max_height), repeat(max_width)))

	frames_body = np.zeros((nb_frames, max_width, max_height), dtype='uint16')
	frames_pharynx = np.zeros(
		(nb_frames, max_width, max_height), dtype='uint16')

	for i, box in enumerate(boxes):
		ymin, xmin = box[0]
		for c in [0, 1]:
			image = video_numpy[i, c, :, :]
			cropped_image = image[ymin:ymin+box[1],
								  xmin:xmin + box[2]].astype('uint16')
			if c == 0:
				frames_pharynx[i, 0:cropped_image.shape[0],
							   0:cropped_image.shape[1]] = cropped_image
			else:
				frames_body[i, 0:cropped_image.shape[0],
							0:cropped_image.shape[1]] = cropped_image

	base = os.path.splitext(os.path.basename(file))[0]
	savename_body = os.path.join(os.path.join(output_dir, 'cropped_body_videos'), base + '_body.tiff')
	tifffile.imwrite(savename_body, frames_body, compression="zlib")

	savename_pharynx = os.path.join(os.path.join(output_dir, 'cropped_pharynx_videos'), base + '_pharynx.tiff')
	tifffile.imwrite(savename_pharynx, frames_pharynx, compression="zlib")

	already_saw_files.append(file)
	already_saw_files.append(savename_body)
	already_saw_files.append(savename_pharynx)

	logging.info(f'Done with : {file}')
	logging.info(f'Execution time : {time() - crop_time} s')


if __name__ == '__main__':
	start_time = time()
	files = sorted([os.path.join(folder_path, x) for x in os.listdir(folder_path)])
	# files = ["/mnt/external.data/TowbinLab/kstojanovski/20220629_Ti2_20x_160-182-190_pumping_25C_20220629_154238_325/raw/Time00242_Point0002_Channel575 nm_Seq12427.nd2"]
	for file in files:
		crop_pharynx(file)
	logging.info(f'Execution time : {time() - start_time} s')

