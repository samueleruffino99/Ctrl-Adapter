import os
import cv2
import numpy as np


def adjust_fov(image, fov_from, fov_to):
    """
    Adjust the Field of View (FOV) of an image from fov_from to fov_to.
    
    Parameters:
    image (numpy.ndarray): The input image.
    fov_from (float): The initial field of view in degrees.
    fov_to (float): The desired field of view in degrees.
    
    Returns:
    numpy.ndarray: The image with adjusted FOV.
    """
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

def process_images(input_dir, output_dir, fov_from, fov_to, adjust_fov_method=True):
    """
    Process images in the input directory by adjusting their FOV and save to the output directory.
    
    Parameters:
    input_dir (str): The directory containing input images.
    output_dir (str): The directory to save the processed images.
    fov_from (float): The initial field of view in degrees.
    fov_to (float): The desired field of view in degrees.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        try:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)
                
                if image is not None:
                    output_path = os.path.join(output_dir, filename)

                    # check if the file already exists
                    if os.path.exists(output_path):
                        # print("File already exists")
                        continue

                    if adjust_fov_method:
                        adjusted_image = adjust_fov(image, fov_from, fov_to)
                    else:
                        adjusted_image = image
                    
                    # Resize the width to 640 pixels
                    height, width = adjusted_image.shape[:2]
                    new_height = int(height * (640 / width))
                    resized_image = cv2.resize(adjusted_image, (640, new_height))
                    
                    # Crop the height to the center 360 pixels
                    start_y = (new_height - 360) // 2
                    cropped_image = resized_image[start_y:start_y+360, :]
                    
                    cv2.imwrite(output_path, cropped_image)
                    print(f"Processed and saved: {output_path}")
                else:
                    print(f"Failed to read image: {image_path}")
        except Exception as e:
            print(e)


import numpy as np
from PIL import Image
import os

# Define the neutral color and thresholds for other colors



# Function to map RGB colors
def map_rgb(image, name):
    img_array = np.array(image)

    if name == "odise":
        neutral_color = (84, 1, 68) 
        thresholds = {
            'sky': ([128, 4, 252], 80, neutral_color),
            'car': ([255, 255, 255], 80, (71, 192, 110)),
            'truck': ([1, 255, 1], 80, (71, 192, 110)),
            'road_markings': ([255, 255, 0], 80, (255, 255, 255)),
            'human': ([0, 0, 255], 80, (71, 30, 112)),
            'road': ([126, 126, 126], 80, (255, 255, 255))
        }
    else:
        neutral_color = (6, 230, 230)
        thresholds = {
            'sky': ([128, 4, 252], 80, neutral_color),
            'car': ([255, 255, 255], 80, (0, 102, 200)),
            'truck': ([1, 255, 1], 80, (255, 0, 20)),
            'road_markings': ([255, 255, 0], 80, (140, 140, 140)),
            'human': ([0, 0, 255], 80, (150, 5, 61)),
            'road': ([126, 126, 126], 80, (140, 140, 140))
        }
    
    # Initialize the mask with the neutral color
    mapped_img_array = np.full(img_array.shape, neutral_color, dtype=np.uint8)

    for key, (target_color, tolerance, new_color) in thresholds.items():
        mask = np.all(np.abs(img_array - target_color) <= tolerance, axis=-1)
        mapped_img_array[mask] = new_color

    return Image.fromarray(mapped_img_array)




def create_video_from_frames(frame_dir, output_video_path, frame_rate, video_length):
    # Get the list of frames
    frames = [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    frames = frames[:33] if len(frames) > 33 else frames
    if "frame" in frames[0]:
        frames.sort(key=lambda x: int(x.split("frame")[-1].replace(".png", "")))
    else:
        frames.sort()  # Ensure frames are in the correct order

    print(f"Total frames found: {len(frames)}")

    total_frames = int(frame_rate * video_length)
    num_frames = len(frames)

    if num_frames < total_frames:
        raise ValueError(f"Not enough frames ({num_frames}) for the desired video length ({video_length}s) and frame rate ({frame_rate}fps)")

    # Calculate the step size for sampling frames if necessary
    if num_frames > total_frames:
        step = num_frames / total_frames
        sampled_frames = [frames[int(i * step)] for i in range(total_frames)]
    else:
        sampled_frames = frames

    print(f"Total frames used for video: {len(sampled_frames)}")

    # Get the size of the frames
    frame = cv2.imread(sampled_frames[0])
    if frame is None:
        raise ValueError(f"Could not read the first frame: {sampled_frames[0]}")
    height, width, layers = frame.shape
    print(f"Frame size: {width}x{height}")

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    if not video.isOpened():
        raise IOError(f"VideoWriter not opened. Output path: {output_video_path}")

    for frame_path in sampled_frames:
        frame = cv2.imread(frame_path)
        # resize the frame to the desired size


        if frame is None:
            raise ValueError(f"Could not read frame: {frame_path}")
        video.write(frame)

    video.release()
    print(f"Video saved to: {output_video_path}")





