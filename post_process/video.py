import os
import cv2
import numpy as np

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


dirs = ["left", "right", "collision", "straight"]
# types = ["rgb", "seg"]
dirs = [ "straight","left", "right", "collision"]
types = ["rgb" ]
for dir in dirs: 
    for type in types:
        try:
        
            base_path = f"/mnt/d/delete/{dir}/"

            output_dir = os.path.join(base_path, f"{type}")

            frame_rate = 10  # frames per second
            video_length = 1.8  # seconds
            output_video_path = os.path.join(base_path, f"output_video_{type}.avi")

            # Ensure the output directory is writable
            if not os.access(base_path, os.W_OK):
                print(f"Cannot write to directory: {base_path}")
            else:
                create_video_from_frames(output_dir, output_video_path, frame_rate, video_length)
        except Exception as e:
            print(e)
