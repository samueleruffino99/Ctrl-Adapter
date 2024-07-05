# Description: This script prepares the data for training the model, generating:
# - csv file with the captions for each scene
# - video segments from the original videos of length segment_length
import os
import json
import cv2
import argparse

from utils.utils_data import get_video_files, get_frames, save_captions

def load_captions(json_path, scene_dir):
    """ Load captions from a json file and match them with the corresponding scene names.
    Args:
        json_path (str): The path to the json file containing the captions.
        scene_dir (str): The directory containing the scenes.
    Returns:
        list: A list of scene names and their corresponding captions. """
    with open(json_path) as f:
        samples = json.load(f)
    scene_names = sorted(os.listdir(scene_dir), key=lambda x: int(x.split('-')[-1].split("_")[0]))
    captions = []
    for scene_name in scene_names:
        scene_name_clean = scene_name.split("_")[0]
        try:
            scene_index = samples["scene_name"].index(scene_name_clean)
            caption = samples["caption"][scene_index][0]
            captions.append([scene_name, caption])
        except ValueError:
            os.remove(os.path.join(scene_dir, scene_name))
            print(f"Scene {scene_name} not found in the json file")
    return captions


def get_segments(frames, segment_length=16):
    """ Get segments of frames.
    Args:
        frames (list): A list of frames.
        segment_length (int): The length of each segment.
    Returns:
        list: A list of segments of frames. """
    segments = [frames[i:i+segment_length] for i in range(0, len(frames), segment_length) if len(frames[i:i+segment_length]) == segment_length]
    return segments


def save_segments(segments, output_dir, video_name):
    """ Save segments of frames as videos.
    Args:
        segments (list): A list of segments of frames.
        output_dir (str): The directory to save the videos.
        video_name (str): The name of the video.
    Returns:
        None """
    for i, segment in enumerate(segments):
        segment_path = os.path.join(output_dir, f"{video_name}_{i}.mp4")
        frame_height, frame_width = segment[0].shape[:2]
        out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
        for frame in segment:
            out.write(frame)
        out.release()

def process_videos(input_dir, output_dir):
    """ Process videos in a directory.
    Args:
        input_dir (str): The directory containing the videos.
        output_dir (str): The directory to save the video segments.
    Returns:
        None """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_files = get_video_files(input_dir)
    for video_file in video_files:
        frames = get_frames(video_file)
        segments = get_segments(frames)
        save_segments(segments, output_dir, os.path.splitext(os.path.basename(video_file))[0])
        print(f"Processed {video_file}")

def main():
    # TODO: ask Jasper for the jappie_seg json file and frames directory data
    parser = argparse.ArgumentParser(description='Prepare Nuscenes data for training the model.')
    parser.add_argument('--input-dir', default='', type=str, help='Directory containing the videos.')
    parser.add_argument('--output-dir', default='', type=str, help='Directory to save the video segments.')
    parser.add_argument('--json-path', default='', type=str, help='Path to the json file containing the captions.')
    parser.add_argument('--csv-filename', default='', type=str, help='Filename of the csv file containing the captions.')
    parser.add_argument('--segment-length', default=16, type=int, help='Length of each video segment.')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugpy for remote debugging')
    args = parser.parse_args()

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        
    # input_dir = "/mnt/e/13_Jasper_diffused_samples/nuscene_videos"
    # output_dir = "/mnt/e/13_Jasper_diffused_samples/nuscene_videos_segments"
    # json_path = "/home/wisley/custom_diffusers_library/src/diffusers/jasper/jappie_seg.json"
    # csv_path = "jasper_captions.csv"

    captions = load_captions(args.json_path, args.output_dir)
    save_captions(captions, args.csv_filename)
    process_videos(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()