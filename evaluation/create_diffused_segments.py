import os
import cv2

def generate_video_segments(folder_path, segment_length, fps, shape):
    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    image_files.sort()

    # Sort the image files in ascending order
    image_files.sort()

    # Calculate the number of segments
    num_segments = len(image_files) // segment_length

    # Create a new folder to store the video segments
    os.makedirs('video_segments', exist_ok=True)

    # Generate video segments
    for i in range(num_segments):

        # Create a new video writer
        video_writer = cv2.VideoWriter(f'video_segments/segment_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, shape)

        # Write the images to the video segment
        for j in range(segment_length):
            image_path = os.path.join(folder_path, image_files[i * segment_length + j])
            image = cv2.imread(image_path)
            video_writer.write(image)

        # Release the video writer
        video_writer.release()

    print(f'{num_segments} video segments generated successfully.')

# Specify the folder path and segment length
folder_path = '/mnt/d/nuscenes/prescanros2_diffused_video/scene-0061'
segment_length = 16
fps = 12
shape = (640, 360)

import debugpy
debugpy.listen(5678)
print("Waiting for debugger attach")
debugpy.wait_for_client()

# Generate video segments
generate_video_segments(folder_path, segment_length, fps, shape)