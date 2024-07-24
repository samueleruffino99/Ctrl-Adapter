# Description: This script prepares the data for training the model, generating:
# - csv file with the captions for each scene
# - video segments from the original videos of length segment_length
import os
import json
import cv2
import argparse
from tqdm import tqdm
import random

from nuscenes.nuscenes import NuScenes

from utils.utils_data import save_captions

# fix random seed for reproducibility
random.seed(0)


def generate_captions(nusc_devkit, json_path, fixed_caption, segments_path, augment_captions=False):
    """ Generate captions for each scene and save them to a csv file.
    Args:
        nusc_devkit (NuScenes): NuScenes dataset instance.
        json_path (str): The path to the json file containing the ANNOTATIONS FROM MLLM.
        fixed_caption (str): Fixed caption for all scenes.
        segments_path (str): The path to the directory containing the video segments.
        augment_captions (bool): Whether to augment the captions with additional information.
    Returns:
        captions (list): A list of scene names and their corresponding captions. """
    # Load annotations from the json file
    with open(json_path) as f:
        annos = json.load(f)
    # Get the scene names from the segments directory
    segments_names = os.listdir(segments_path)
    # Generate captions for each scene segment
    captions = []
    for scene_name in nusc_devkit.scene:
        scene_token = scene_name['token']
        anno = annos[scene_token]
        for segment_name in segments_names:
            if scene_name['name'] in segment_name:
                caption = fixed_caption
                if augment_captions and random.random() > 0.5:
                    scenary_conditions = anno.get('scenary', {})
                    environmental_conditions = anno.get('environmental_condition', {})
                    # lightning condition
                    if random.random() > 0.5:
                        caption += f", during {environmental_conditions['lighting']} light"
                    # weather condition
                    if random.random() > 0.5:
                        caption += f", in {environmental_conditions['weather']} weather"
                    # road type and conditions
                    if random.random() > 0.5:
                        caption += f", on a {environmental_conditions['road_conditions']} {environmental_conditions['road_type']} road"
                    # scene type conditions
                    if random.random() > 0.5:
                        caption += f", in an {scenary_conditions['scene_type']} environment"
                    # traffic conditions
                    if random.random() > 0.5:
                        caption += f", with {scenary_conditions['traffic_conditions']} traffic"
                # add '.'
                caption += '.'
                captions.append([segment_name, caption])
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


def save_segments(segments, output_dir, scene_name):
    """ Save segments of frames as videos.
    Args:
        segments (list): A list of segments of frames.
        output_dir (str): The directory to save the videos.
        scene_name (str): The name of the scene.
    Returns:
        None """
    for i, segment in enumerate(segments):
        segment_path = os.path.join(output_dir, f"{scene_name}_{i}.mp4")
        frame_height, frame_width = segment[0].shape[:2]
        out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
        for frame in segment:
            out.write(frame)
        out.release()

def process_videos(input_dir, output_dir, segment_length=16):
    """ Process videos in a directory.
    Args:
        input_dir (str): The directory containing all the scenes folders with frames.
        output_dir (str): The directory to save the video segments.
        segment_length (int): The length of each segment.
    Returns:
        None """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scene_list = os.listdir(input_dir)
    for scene in tqdm(scene_list):
        scene_dir = os.path.join(input_dir, scene)
        frames = os.listdir(scene_dir)
        frames = [cv2.imread(os.path.join(scene_dir, frame)) for frame in frames]
        segments = get_segments(frames, segment_length)
        save_segments(segments, output_dir, scene)

def main():
    parser = argparse.ArgumentParser(description='Prepare Nuscenes data for training the model.')
    parser.add_argument('--dataroot', type=str, default='/mnt/d/nuscenes', help='Path to Nuscenes dataset.')
    parser.add_argument('--version', type=str, default='v1.0-trainval', help='Nuscenes dataset version.')
    parser.add_argument('--json-filename', default='results_nusc_mllm.json', type=str, help='Filename of the json file containing the annotation from mllm LLaVA.')
    parser.add_argument('--input-dir', default='scenes_frames', type=str, help='Folder containing the frames per each scene.')
    parser.add_argument('--output-dir', default='scenes_videos_segments', type=str, help='folder to save the video segments.')
    parser.add_argument('--cam-type', type=str, default='CAM_FRONT', help='Camera type to extract images.')
    parser.add_argument('--use-adjusted-fov', default=True, action='store_true', help='Use adjusted field of view.')
    parser.add_argument('--generate-segments', default=False, action='store_true', help='Generate video segments.')
    parser.add_argument('--fixed-caption', default='A realistic driving scene', type=str, help='Fixed caption for all scenes.')
    parser.add_argument('--augment-captions', default=False, action='store_true', help='Augment captions with additional information.')
    parser.add_argument('--csv-filename', default='video_captions_nuscenes.csv', type=str, help='Filename of the csv file containing the captions.')
    parser.add_argument('--segment-length', default=16, type=int, help='Length of each video segment.')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugpy for remote debugging')
    args = parser.parse_args()
    cam_type_foldername = args.cam_type + "_adj_fov" if args.use_adjusted_fov else args.cam_type
    args.input_path = os.path.join(args.dataroot, 'scenes_frames', cam_type_foldername)
    args.output_path = os.path.join(args.dataroot, args.output_dir)
    args.json_path = os.path.join(args.dataroot, args.version, 'predictions', 'mllm', args.json_filename)
    args.csv_path = os.path.join('sample_data', args.csv_filename)

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    nusc_devkit = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Generate video segments
    if args.generate_segments:
        process_videos(args.input_path, args.output_path, args.segment_length)

    # Generate captions from a fixed caption plus additional information from the json file
    captions = generate_captions(nusc_devkit, args.json_path, args.fixed_caption, args.output_path, args.augment_captions)
    save_captions(captions, args.csv_path)

if __name__ == "__main__":
    main()