# Description: This script prepares the data for training the model, generating:
# - csv file with the captions for each scene
# - video segments from the original videos of length segment_length
import os
import json
import cv2
import argparse
from tqdm import tqdm

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


def generate_captions(fixed_caption, segments_path, csv_path):
    """ Generate captions for each scene and save them to a csv file.
    Args:
        fixed_caption (str): Fixed caption for all scenes.
        segments_path (str): The path to the directory containing the video segments.
        csv_path (str): The path to the csv file.
    Returns:
        captions (list): A list of scene names and their corresponding captions. """
    scene_names = os.listdir(segments_path)
    captions = [[scene_name, fixed_caption] for scene_name in scene_names]
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
    # TODO: ask Jasper for the jappie_seg.json file and frames directory data
    parser = argparse.ArgumentParser(description='Prepare Nuscenes data for training the model.')
    parser.add_argument('--dataroot', type=str, default='/mnt/d/nuscenes', help='Path to Nuscenes dataset.')
    parser.add_argument('--input-dir', default='scenes_frames', type=str, help='Folder containing the frames per each scene.')
    parser.add_argument('--output-dir', default='scenes_videos_segments', type=str, help='folder to save the video segments.')
    parser.add_argument('--cam-type', type=str, default='CAM_FRONT', help='Camera type to extract images.')
    parser.add_argument('--use-adjusted-fov', default=True, action='store_true', help='Use adjusted field of view.')
    parser.add_argument('--load-captions', default=False, action='store_true', help='Load captions from a json file.')
    parser.add_argument('--json-filename', default='captions.json', type=str, help='Path to the json file containing the captions, if present.')
    parser.add_argument('--fixed-caption', default='A driving scene.', type=str, help='Fixed caption for all scenes.')
    parser.add_argument('--csv-filename', default='video_captions_nuscenes.csv', type=str, help='Filename of the csv file containing the captions.')
    parser.add_argument('--segment-length', default=16, type=int, help='Length of each video segment.')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugpy for remote debugging')
    args = parser.parse_args()
    cam_type_foldername = args.cam_type + "_adj_fov" if args.use_adjusted_fov else args.cam_type
    args.input_path = os.path.join(args.dataroot, 'scenes_frames', cam_type_foldername)
    args.output_path = os.path.join(args.dataroot, args.output_dir)
    args.json_path = os.path.join(args.dataroot, args.json_filename)
    args.csv_path = os.path.join('sample_data', args.csv_filename)

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # input_dir = "/mnt/e/13_Jasper_diffused_samples/nuscene_videos"
    # output_dir = "/mnt/e/13_Jasper_diffused_samples/nuscene_videos_segments"
    # json_path = "/home/wisley/custom_diffusers_library/src/diffusers/jasper/jappie_seg.json"
    # csv_path = "jasper_captions.csv"

    # Generate video segments
    process_videos(args.input_path, args.output_path, args.segment_length)

    # Load captions from a json file and save them to a csv file (if json present, otherwise use fixed caption)
    if args.load_captions:
        captions = load_captions(args.json_path, args.input_path)
    else:
        captions = generate_captions(args.fixed_caption, args.output_path, args.csv_filename)
    save_captions(captions, args.csv_path)

if __name__ == "__main__":
    main()