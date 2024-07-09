# Description: Script to prepare front view images from nuScenes dataset, generating a folder with the resized front-view frames for each scene.
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from nuscenes.nuscenes import NuScenes
from utils.utils_data import *


def process_nuscenes_data(nusc, data_dir, scale_factor=0.5, cam_type="CAM_FRONT", save_adjusted_fov=False, fov_from=120, fov_to=94):
    """
    Process each scene in the nuScenes dataset to extract, resize, and save front view images.

    Args:
        nusc (NuScenes): nuScenes dataset instance.
        data_dir (str): Directory to save the processed images.
        scale_factor (float): Scale factor for resizing images.
        cam_type (str): Camera type to extract images.
        save_adjusted_fov (bool): Save images with adjusted field of view.
        fov_from (float): Initial field of view in degrees.
        fov_to (float): Desired field of view in degrees.
    """
    data_dir_orig = os.path.join(data_dir, cam_type)
    if not os.path.exists(data_dir_orig):
        os.makedirs(data_dir_orig)

    if save_adjusted_fov:
        data_dir_adj_fov = os.path.join(data_dir, f"{cam_type}_adj_fov")
        if not os.path.exists(data_dir_adj_fov):
            os.makedirs(data_dir_adj_fov)

    
    for scene in tqdm(nusc.scene):
        scene_name = scene['name']
        scene_dir = os.path.join(data_dir_orig, scene_name)

        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)

        if save_adjusted_fov:
            scene_dir_adj_fov = os.path.join(data_dir_adj_fov, scene_name)
            if not os.path.exists(scene_dir_adj_fov):
                os.makedirs(scene_dir_adj_fov)

        first_sample_token = scene['first_sample_token']
        sample_record = nusc.get('sample', first_sample_token)
        sample = nusc.get('sample_data', sample_record['data'][cam_type])
        
        i = 0 
        while sample:
            image_filepath = nusc.get_sample_data_path(sample['token'])
            image = cv2.imread(image_filepath)

            # Resize original image and save
            resized_image = resize_image(image, scale_factor)
            cv2.imwrite(os.path.join(scene_dir, f"{i:06d}.png"), resized_image)

            # Change fov of resized image if needed and save
            if save_adjusted_fov:
                resized_image_adj_fov = adjust_fov(resized_image, fov_from, fov_to)
                cv2.imwrite(os.path.join(scene_dir_adj_fov, f"{i:06d}.png"), resized_image_adj_fov)
            
            # Move to next sample
            if sample['next'] == '':
                break
            sample = nusc.get('sample_data', sample['next'])
            i += 1

# Main function to setup nuScenes SDK and path
def main():
    parser = argparse.ArgumentParser(description='Generate front view images from nuScenes dataset')
    parser.add_argument('--dataroot', type=str, default="/mnt/d/nuscenes", help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default="v1.0-trainval", help='nuScenes dataset version')
    parser.add_argument('--cam-type', type=str, default="CAM_FRONT", help='Camera type to extract images')
    parser.add_argument('--output-dir', type=str, default="scenes_frames", help='Path to output directory')
    parser.add_argument('--scale-factor', type=float, default=0.4, help='Scale factor for resizing images, 0.4 correspond to divide by 2.5 each dimension.')
    parser.add_argument('--save-adjusted-fov', action='store_true', help='Save images with adjusted field of view')
    parser.add_argument('--fov-from', type=float, default=120, help='Initial field of view in degrees')
    parser.add_argument('--fov-to', type=float, default=94, help='Desired field of view in degrees (94 is to match the Presan segmentation fov)')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugpy for remote debugging')
    args = parser.parse_args()

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    args.output_path = os.path.join(args.dataroot, args.output_dir)
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    process_nuscenes_data(nusc, args.output_path, args.scale_factor, args.cam_type, args.save_adjusted_fov, args.fov_from, args.fov_to)

if __name__ == "__main__":
    main()
