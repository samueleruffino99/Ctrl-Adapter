import os
import cv2
import numpy as np
from PIL import Image
import csv


def get_video_files(path):
    """ Get a list of video files in a directory.
    Args:
        path (str): The path to the directory.
    Returns:
        list: A list of video files."""
    video_files = []
    for root, dirs, files in os.walk(path):
        if files:
            video_files.extend([os.path.join(root, file) for file in files if file.endswith(".avi")])
    return video_files


def get_frames(video_file):
    """ Get the frames of a video.
    Args:
        video_file (str): The path to the video file.
    Returns:
        list: A list of frames."""
    cap = cv2.VideoCapture(video_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_captions(captions, csv_path):
    """ Save captions to a csv file.
    Args:
        captions (list): A list of scene names and their corresponding captions.
        csv_path (str): The path to the csv file.
    Returns:
        None """
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "caption"])
        writer.writerows(captions)


def adjust_fov(image, fov_from=120, fov_to=94):
    """
    Adjust the Field of View (FOV) of an image from fov_from to fov_to.
    Args:
        image (numpy.ndarray): The input image.
        fov_from (float): The initial field of view in degrees.
        fov_to (float): The desired field of view in degrees.
    Returns:
        numpy.ndarray: The image with adjusted FOV. """
    height, width = image.shape[:2]
    
    # Convert degrees to radians
    fov_from_rad = np.deg2rad(fov_from)
    fov_to_rad = np.deg2rad(fov_to)
    
    # Calculate the focal lengths
    focal_length_from = 0.5 * width / np.tan(fov_from_rad / 2)
    focal_length_to = 0.5 * width / np.tan(fov_to_rad / 2)
    
    # Calculate the scaling factor
    fov_scale = focal_length_to / focal_length_from
    
    # Intrinsic camera matrix
    K = np.array([[width, 0, width / 2],
                  [0, width, height / 2],
                  [0, 0, 1]], dtype=np.float32)
    
    # Adjust the camera matrix
    K_new = K.copy()
    K_new[0, 0] *= fov_scale
    K_new[1, 1] *= fov_scale
    
    map1, map2 = cv2.initUndistortRectifyMap(K, np.zeros(5), None, K_new, (width, height), 5)
    dst = cv2.remap(image, map1, map2, cv2.INTER_LINEAR)
    
    return dst

def resize_image(image, scale_factor=0.5):
    """ Resize an image by a scale factor.
    Args:
        image (numpy.ndarray): The input image.
        scale_factor (float): The scale factor.
    Returns:
        numpy.ndarray: The resized image. """
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    return cv2.resize(image, (new_width, new_height))


def map_rgb(image, mapping_type=None):
    """ Map prescan segmentation map colors to a predefined set of colors.
    Args:
        image (PIL.Image): The input image.
        mapping_type (str): The type of mapping to apply. It can be 'odise', 'ade' or None (no mapping).
    Returns:
        PIL.Image: The mapped image. """

    if mapping_type == "odise":
        neutral_color = (84, 1, 68) 
        thresholds = {
            'sky': ([128, 4, 252], 80, neutral_color),
            'car': ([255, 255, 255], 80, (71, 192, 110)),
            'truck': ([1, 255, 1], 80, (71, 192, 110)),
            'road_markings': ([255, 255, 0], 80, (255, 255, 255)),
            'human': ([0, 0, 255], 80, (71, 30, 112)),
            'road': ([126, 126, 126], 80, (255, 255, 255))
        }
    elif mapping_type == "ade":
        neutral_color = (6, 230, 230)
        thresholds = {
            'sky': ([128, 4, 252], 80, neutral_color),
            'car': ([255, 255, 255], 80, (0, 102, 200)),
            'truck': ([1, 255, 1], 80, (255, 0, 20)),
            'road_markings': ([255, 255, 0], 80, (140, 140, 140)),
            'human': ([0, 0, 255], 80, (150, 5, 61)),
            'road': ([126, 126, 126], 80, (140, 140, 140))
        }
    else:
        return image
    
    img_array = np.array(image)

    # Initialize the mask with the neutral color
    mapped_img_array = np.full(img_array.shape, neutral_color, dtype=np.uint8)

    for key, (target_color, tolerance, new_color) in thresholds.items():
        mask = np.all(np.abs(img_array - target_color) <= tolerance, axis=-1)
        mapped_img_array[mask] = new_color

    return Image.fromarray(mapped_img_array)

