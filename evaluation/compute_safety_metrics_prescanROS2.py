import torch
import torch.utils
import numpy as np
import os
import PIL.Image as Image
import argparse
import cv2
from scipy.stats import ks_2samp
import json
import matplotlib.pyplot as plt

from ultralytics import YOLO, YOLOWorld, solutions

# define 
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
AREA_THRESHOLDS = {
    'person': 2000,
    'bycicle': 2000,
    'car': 10000,
    'motorcycle': 2000,
    'bus': 10000,
    'truck': 10000,
    'traffic light': 1500,
    'fire hydrant': 1500,
    'stop sign': 1500,
    'bench': 1500,
    'cat': 1000,
    'dog': 1000,
    'potted plant': 2000
}
AREA_CHNAGE_THRESHOLD = 1.50


def load_YOLO():
    """Load YOLOv8 model from Ultralytics (detection)."""
    model = YOLO('yolov8n.pt')
    return model 


class VideoDataset(torch.utils.data.Dataset):
    """Dataset class for FID calculation."""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image

    
def generate_detection_results(model, image_paths, orig, save_vis):
    """Generate YOLO results for a list of images.
    Args:
        model (torch.nn.Module): Model to use for evaluation.
        image_paths (list): List of image paths.
        orig (bool): Whether the images are original or generated.
        save_vis (bool): Whether to save visualizations.
    Returns:
        list: List of YOLO results for each image.
    """
    det_list = []
    scene_name = os.path.basename(os.path.dirname(image_paths[0]))
    results = model(image_paths, verbose=False, save=save_vis, project=f"evaluation/safety_metrics/{scene_name}/{'original' if orig else 'generated'}", name='detections')
    for i, result in enumerate(results):
        det_dict = {}
        det_dict['conf'] = result.boxes.conf.cpu().numpy()
        mask = det_dict['conf'] > CONFIDENCE_THRESHOLD
        det_dict['boxes'] = result.boxes.xyxy[mask].cpu().numpy()
        det_dict['cls'] = result.boxes.cls[mask].cpu().numpy()
        det_dict['conf'] = result.boxes.conf[mask].cpu().numpy()
        det_dict['area'] = (det_dict['boxes'][:, 2] - det_dict['boxes'][:, 0]) * (det_dict['boxes'][:, 3] - det_dict['boxes'][:, 1])
        # compute criticality based on area 
        det_dict['criticality'] = []
        for area, cls in zip(det_dict['area'], det_dict['cls']):
            if model.names[cls] not in AREA_THRESHOLDS:
                det_dict['criticality'].append(0)
                continue
            if area > AREA_THRESHOLDS[model.names[cls]]:
                det_dict['criticality'].append(1)
            else:
                det_dict['criticality'].append(0)
        det_list.append(det_dict)
    return det_list


def generate_track_and_heatmap_results(model, image_paths, orig, save_vis):
    """Generate YOLO heatmap results for a list of images.
    Args:
        model (torch.nn.Module): Model to use for evaluation.
        image_paths (list): List of image paths.
        orig (bool): Whether the images are original or generated.
        save_vis (bool): Whether to save visualizations.
    """
    scene_name = os.path.basename(os.path.dirname(image_paths[0]))
    output_path = f"evaluation/safety_metrics/{scene_name}/{'original' if orig else 'generated'}/heatmap"
    # Init heatmap
    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_CIVIDIS,
        view_img=False,
        imw=640,
        imh=360,
        names=model.names,
    )
    tracks = model.track(image_paths, persist=True, show=False, verbose=False, save=save_vis, project=f"evaluation/safety_metrics/{scene_name}/{'original' if orig else 'generated'}", name='tracks')
    track_list = []
    for i, (image_path, track) in enumerate(zip(image_paths, tracks)):
        # generate track results
        track_dict = {}
        track_dict['conf'] = track.boxes.conf.cpu().numpy()
        mask = track_dict['conf'] > CONFIDENCE_THRESHOLD
        track_dict['boxes'] = track.boxes.xyxy[mask].cpu().numpy()
        track_dict['cls'] = track.boxes.cls[mask].cpu().numpy()
        track_dict['conf'] = track.boxes.conf[mask].cpu().numpy()
        if track.boxes.id is not None:
            track_dict['track_ids'] = track.boxes.id[mask].cpu().numpy()
        else:
            track_dict['track_ids'] = None
        track_dict['area'] = (track_dict['boxes'][:, 2] - track_dict['boxes'][:, 0]) * (track_dict['boxes'][:, 3] - track_dict['boxes'][:, 1])
        # compute criticality based on area 
        track_dict['criticality'] = []
        for area, cls in zip(track_dict['area'], track_dict['cls']):
            if model.names[cls] not in AREA_THRESHOLDS:
                track_dict['criticality'].append(0)
                continue
            if area > AREA_THRESHOLDS[model.names[cls]]:
                track_dict['criticality'].append(1)
            else:
                track_dict['criticality'].append(0)
        track_list.append(track_dict)
        # generate heatmap visualization
        if save_vis:
            img = cv2.imread(image_path)
            if track.boxes.shape[0] == 0:
                track.boxes.data = [track.boxes.data]
            img = heatmap_obj.generate_heatmap(img, track)
            os.makedirs(output_path, exist_ok=True)
            cv2.imwrite(f"{output_path}/{i:06d}.png", img)
    return track_list


def iou(box1, box2):
    """Compute Intersection over Union (IoU) between two boxes.
    Args:
        box1 (list): Box coordinates [x1, y1, x2, y2].
        box2 (list): Box coordinates [x1, y1, x2, y2].
    Returns:
        float: IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # compute the area of input bboxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Compute the union area
    union_area = box1_area + box2_area - intersection_area

    # Compute the intersection over union 
    iou = intersection_area / float(union_area)
    return iou


def compute_undetected_objects(tracks_orig, tracks_gen, debug=False):
    """Compute undetected objects between two sets of tracks.
    Args:
        tracks_orig (list): List of tracks for original data.
        tracks_gen (list): List of tracks for generated data.
        debug (bool): Whether to print debug information.
    Returns:
        count_undetected_objects_all (int): number of undetected objects (all)
        count_undetected_objects_critical (int): number of undetected objects (critical --> high area)
        count_undetected_objects_noncritical (int): number of undetected objects (non critical --> low area)
    """
    count_objects_orig_all, count_undetected_objects_all, count_undetected_objects_critical, count_undetected_objects_noncritical = 0, 0, 0, 0
    for orig, gen in zip(tracks_orig, tracks_gen):
        # compute IoU between each pair of boxes
        iou_matrix = np.zeros((len(orig['boxes']), len(gen['boxes'])))
        for i, orig_box in enumerate(orig['boxes']):
            for j, gen_box in enumerate(gen['boxes']):
                iou_matrix[i, j] = iou(orig_box, gen_box)
        # find the best match for each original box
        for i in range(len(orig['boxes'])):
            count_objects_orig_all += 1
            best_match = np.argmax(iou_matrix[i])
            if iou_matrix[i, best_match] < IOU_THRESHOLD:
                count_undetected_objects_all += 1
                if orig['criticality'][i] == 1:
                    count_undetected_objects_critical += 1
                else:
                    count_undetected_objects_noncritical += 1
    false_negative_rate = count_undetected_objects_all / count_objects_orig_all
    false_negative_rate_critical = count_undetected_objects_critical / count_objects_orig_all
    false_negative_rate_noncritical = count_undetected_objects_noncritical / count_objects_orig_all
    if debug:
        print("========================== Undetection Metrics ==========================")
        print(f"Total number of detections in original video: {count_objects_orig_all}")
        print(f"Total number of undetected objects in generated video: {count_undetected_objects_all}")
        print(f"Total number of undetected critical objects in generated video (area > threshold): {count_undetected_objects_critical}")
        print(f"Total number of undetected non-critical objects in generated video (area < threshold): {count_undetected_objects_noncritical}")
        print(f"False negative rate (all): {false_negative_rate}")
        print(f"False negative rate (critical): {false_negative_rate_critical}")
        print(f"False negative rate (non-critical): {false_negative_rate_noncritical}")
        print("=========================================================================")
    return count_objects_orig_all, count_undetected_objects_all, count_undetected_objects_critical, count_undetected_objects_noncritical


def compute_area_changes(tracks, segments_length, class_names):
    """
    Calculate the increase in area for each track over a specified number of frames.
    Args:
        tracks (list): List of tracks, where each track contains area and track ID information.
        segments_length (int): The number of frames over which to calculate area changes.
        class_names (list): List of class names.
    Returns:
        dict: A dictionary with track IDs and their area increase info and criticality flags.
    """
    track_areas = {}
    track_changes = {}
    # Accumulate areas by track ID over the sequence of frames
    for track in tracks:
        for i, bbox in enumerate(track['boxes']):
            area = track['area'][i]
            criticality = track['criticality'][i]
            class_id = track['cls'][i]
            track_id = None
            if track['track_ids'] is not None:
                track_id = track['track_ids'][i]
            if track_id is not None:
                if track_id not in track_areas:
                    track_areas[track_id] = {'cls': class_id, 'areas': []}
                track_areas[track_id]['areas'].append(area)
    # Calculate changes over the defined segment length
    for track_id, info in track_areas.items():
        areas = info['areas']
        for start_idx in range(0, len(areas), segments_length):
            end_idx = min(start_idx + segments_length, len(areas))
            initial_area = areas[start_idx]
            latest_area = areas[end_idx - 1]
            area_change = latest_area / initial_area if initial_area > 0 else 0
            is_critical = area_change >= AREA_CHNAGE_THRESHOLD
            track_changes.setdefault(track_id, []).append({
                'cls': class_names[info['cls']],
                'initial_area': initial_area,
                'latest_area': latest_area,
                'area_change': area_change,
                'is_area_critical': criticality,
                'is_area_change_critical': is_critical
            })
    return track_changes


def compute_stats(area_changes):
        """Calculate mean, standard deviation, max, and min of area changes.
        Args:
            area_changes (dict): Area change info for a set of tracks.
        Returns:
            tuple: Mean, standard deviation, max, and min of area changes."""
        changes = [change['area_change'] for track in area_changes.values() for change in track]
        mean_change = sum(changes) / len(changes)
        std_changes = (sum((x - mean_change) ** 2 for x in changes) / len(changes)) ** 0.5
        max_change = max(changes)
        min_change = min(changes)
        return changes, mean_change, std_changes, max_change, min_change


def area_change_criticality_by_class(area_changes):
        """Calculate the average criticality of area changes per class.
        Args:
            area_changes (dict): Area change info for a set of tracks.
        Returns:
            dict: Dictionary containing the average criticality per class."""
        class_names = AREA_THRESHOLDS.keys()
        class_criticality = {name: [] for name in class_names}
        for track in area_changes.values():
            for change in track:
                class_name = change['cls']  # Assuming change['cls'] is an index
                if change['is_area_change_critical']:
                    class_criticality[class_name].append(change['area_change'])

        # Calculate average criticality per class
        for name in class_criticality:
            if class_criticality[name]:
                class_criticality[name] = sum(class_criticality[name]) / len(class_criticality[name])
            else:
                class_criticality[name] = 0  # Handle case where no critical events occur for a class
        return class_criticality


def compare_area_change_criticality_distributions(area_changes_orig, area_changes_gen):
        """Compare the criticality distributions of area changes between two sets of tracks.
        Args:
            area_changes_orig (dict): Area change info for original data.
            area_changes_gen (dict): Area change info for generated data.
        Returns:
            tuple: Kolmogorov-Smirnov statistic and p-value."""
        critical_changes_orig = [change['area_change'] for track in area_changes_orig.values() for change in track if change['is_area_change_critical']]
        critical_changes_gen = [change['area_change'] for track in area_changes_gen.values() for change in track if change['is_area_change_critical']]
        
        ks_stat, p_value = ks_2samp(critical_changes_orig, critical_changes_gen)
        return ks_stat, p_value


def compute_area_change_criticality(area_changes_orig, area_changes_gen, debug=False):
    """Compute criticality of area changes between two sets of tracks.
    Args:
        area_changes_orig (dict): Area change info for original data.
        area_changes_gen (dict): Area change info for generated data.
        debug (bool): Whether to print debug information.
    Returns:
        tuple: Results for original data, generated data, and comparison.
    """
    # Calculate stats for original and generated data
    all_orig, mean_orig, std_orig, max_orig, min_orig = compute_stats(area_changes_orig)
    al_gen, mean_gen, std_gen, max_gen, min_gen = compute_stats(area_changes_gen)

    # Results of orginal and generated data
    results_orig = {
        'all_changes': all_orig,
        'mean_change': mean_orig,
        'std_change': std_orig,
        'max_change': max_orig,
        'min_change': min_orig,
        'class_criticality': area_change_criticality_by_class(area_changes_orig)
    }
    results_gen = {
        'all_changes': al_gen,
        'mean_change': mean_gen,
        'std_change': std_gen,
        'max_change': max_gen,
        'min_change': min_gen,
        'class_criticality': area_change_criticality_by_class(area_changes_gen)
    }

    ks_stat, ks_p_value = compare_area_change_criticality_distributions(area_changes_orig, area_changes_gen)
    results_comparison = {
        'mean_change_diff': mean_gen - mean_orig,
        'std_change_diff': std_gen - std_orig,
        'max_change_diff': max_gen - max_orig,
        'min_change_diff': min_gen - min_orig,
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p_value
    }
    # Criticality analysis
    critical_orig = sum(1 for track in area_changes_orig.values() for change in track if change['is_area_change_critical'])
    critical_gen = sum(1 for track in area_changes_gen.values() for change in track if change['is_area_change_critical']) 
    results_orig['critical_changes_percentage'] = 100 * critical_orig / sum(len(track) for track in area_changes_orig.values())
    results_gen['critical_changes_percentage'] = 100 * critical_gen / sum(len(track) for track in area_changes_gen.values())
    # Analysis results
    if debug:
        print("========================== Area Change Metrics ==========================")
        print(f"Mean area change: {mean_orig:.2f}")
        print(f"Standard deviation of area change: {std_orig:.2f}")
        print(f"Max area change: {max_orig:.2f}")
        print(f"Min area change: {min_orig:.2f}")
        print(f"Critical changes percentage: {results_orig['critical_changes_percentage']:.2f}%")
        print("Generated data:")
        print(f"Mean area change: {mean_gen:.2f}")
        print(f"Standard deviation of area change: {std_gen:.2f}")
        print(f"Max area change: {max_gen:.2f}")
        print(f"Min area change: {min_gen:.2f}")
        print(f"Critical changes percentage: {results_gen['critical_changes_percentage']:.2f}%")
        print("Comparison (generated - original):")
        print(f"Mean area change difference: {results_comparison['mean_change_diff']:.2f}")
        print(f"Standard deviation of area change difference: {results_comparison['std_change_diff']:.2f}")
        print(f"Max area change difference: {results_comparison['max_change_diff']:.2f}")
        print(f"Min area change difference: {results_comparison['min_change_diff']:.2f}")
        print(f"Kolmogorov-Smirnov statistic: {results_comparison['ks_statistic']:.2f}")
        print(f"Kolmogorov-Smirnov p-value: {results_comparison['ks_p_value']:.2f}")
        print("=========================================================================")
    return results_orig, results_gen, results_comparison


def create_videos_from_images(scene_name, fps):
    """Create videos from images in a directory.
    Args:
        scene_name (str): Name of the scene.
        fps (int): Frames per second for the output video.
    """
    orig_frames_path = f"evaluation/safety_metrics/{scene_name}/original"
    gen_frames_path = f"evaluation/safety_metrics/{scene_name}/generated"
    output_path = f"evaluation/safety_metrics/{scene_name}"

    # Create video of original and generated frames in a single video:
    # Top row: left - original tracks frame, right - generated tracks frame
    # Bottom row: left - original heatmap frame, right - generated heatmap frame
    orig_tracking_frames = sorted([f for f in os.listdir(os.path.join(orig_frames_path, "tracks")) if f.endswith('.png')])
    gen_tracking_frames = sorted([f for f in os.listdir(os.path.join(gen_frames_path, "tracks")) if f.endswith('.png')])
    orig_heatmap_frames = sorted([f for f in os.listdir(os.path.join(orig_frames_path, "heatmap")) if f.endswith('.png')])
    gen_heatmap_frames = sorted([f for f in os.listdir(os.path.join(gen_frames_path, "heatmap")) if f.endswith('.png')])
    assert len(orig_tracking_frames) == len(gen_tracking_frames) == len(orig_heatmap_frames) == len(gen_heatmap_frames)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame = cv2.imread(os.path.join(orig_frames_path, "tracks", orig_tracking_frames[0]))
    frame_height, frame_width, _ = frame.shape
    out = cv2.VideoWriter(f"{output_path}/comparison.mp4", fourcc, fps, (frame_width * 2, frame_height * 2))
    for i in range(len(orig_tracking_frames)):
        orig_track_frame = cv2.imread(os.path.join(orig_frames_path, "tracks", orig_tracking_frames[i]))
        gen_track_frame = cv2.imread(os.path.join(gen_frames_path, "tracks", gen_tracking_frames[i]))
        orig_heatmap_frame = cv2.imread(os.path.join(orig_frames_path, "heatmap", orig_heatmap_frames[i]))
        gen_heatmap_frame = cv2.imread(os.path.join(gen_frames_path, "heatmap", gen_heatmap_frames[i]))
        top_row = np.hstack((orig_track_frame, gen_track_frame))
        bottom_row = np.hstack((orig_heatmap_frame, gen_heatmap_frame))
        frame = np.vstack((top_row, bottom_row))
        cv2.putText(frame, scene_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()


def plot_safety_metrics_for_scene(scene_metrics):
    """Plot safety metrics for a scene.
    Args:
        scene_metrics (dict): Dictionary containing the calculated metrics for a scene.
    """
    output_path = f"evaluation/safety_metrics/{scene_metrics['scene']}"
    os.makedirs(output_path, exist_ok=True)
    ### Undetected objects
    undetected_objects = scene_metrics['undetected_objects']
    data = [undetected_objects['false_negative_rate'], undetected_objects['false_negative_rate_critical'], undetected_objects['false_negative_rate_noncritical']]
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"Scene: {scene_metrics['scene']}", fontsize=16)
    plt.ylim(0, 1)
    plt.bar(['All', 'Critical', 'Non-critical'], data)
    # add value on top of the bar
    for i, v in enumerate(data):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.ylabel('False Negative Rate')
    plt.savefig(f"evaluation/safety_metrics/{scene_metrics['scene']}/undetected_objects_stats.png")
    plt.close()
    ### Area changes
    area_changes = scene_metrics['area_changes']
    orig_changes = area_changes['original']['all_changes']
    gen_changes = area_changes['generated']['all_changes']
    data = [orig_changes, gen_changes]
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Scene: {scene_metrics['scene']}", fontsize=16)
    # Area changes statistics
    ax[0, 0].set_title('Bounding Boxes Area Changes Statistics')
    ax[0, 0].violinplot(data, showmeans=True)
    ax[0, 0].set_xticks([1, 2])
    ax[0, 0].set_xticklabels(['Original', 'Generated'])
    ax[0, 0].set_ylabel('Area Change')
    # Area changes percentage
    ax[0, 1].set_title('Percentage of Critical Area Changes')
    ax[0, 1].set_ylim(0, 100)
    ax[0, 1].bar(['Original', 'Generated'], [area_changes['original']['critical_changes_percentage'], area_changes['generated']['critical_changes_percentage']])
    ax[0, 1].set_ylabel('Percentage')
    # Criticality per class
    class_criticality_orig = area_changes['original']['class_criticality']
    class_criticality_gen = area_changes['generated']['class_criticality']
    class_names = list(class_criticality_orig.keys())
    class_criticality_orig = [class_criticality_orig[name] for name in class_names]
    class_criticality_gen = [class_criticality_gen[name] for name in class_names]
    x = np.arange(len(class_names))  # the label locations
    width = 0.35  # the width of the bars
    ax[1, 0].set_title('Average Criticality of Area Changes per Class')
    rects1 = ax[1, 0].bar(x - width/2, class_criticality_orig, width, label='Original', alpha=0.5)
    rects2 = ax[1, 0].bar(x + width/2, class_criticality_gen, width, label='Generated', alpha=0.5)
    ax[1, 0].set_ylabel('Average Criticality')
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(class_names, rotation=60)
    ax[1, 0].legend()
    ax[1, 1].axis('off')  # Turn off axis and hide subplot
    plt.tight_layout()
    plt.savefig(f"evaluation/safety_metrics/{scene_metrics['scene']}/area_changes_stats.png")
    plt.close()

    
def calculate_safety_for_scenes(original_dir, generated_dir, segments_length, fps, model, debug=False, save_vis=False):
    """Calculate safety metrics for each scene.
    Args:
        original_dir (str): Path to the original data directory.
        generated_dir (str): Path to the generated data directory.
        segments_length (int): Number of frames per video segment.
        fps (int): Frames per second for the output video.
        model (torch.nn.Module): Model to use for evaluation.
        debug (bool): Whether to print debug information.
    Returns:
        dict: Dictionary containing the calculated metrics for each scene.
    """
    scene_folders = sorted(os.listdir(generated_dir))
    results = {}

    for scene in scene_folders:
        print(f"Processing scene: {scene}")

        # if already eist delete the folder
        if os.path.exists(f"evaluation/safety_metrics/{scene}"):
            os.system(f"rm -rf evaluation/safety_metrics/{scene}")

        original_scene_path = os.path.join(original_dir, scene)
        generated_scene_path = os.path.join(generated_dir, scene)

        original_image_paths = sorted([os.path.join(original_scene_path, f) for f in os.listdir(original_scene_path)])
        generated_image_paths = sorted([os.path.join(generated_scene_path, f) for f in os.listdir(generated_scene_path)])

        # run YOLO detection on all images
        # det_orig = generate_detection_results(model, original_image_paths, True, save_vis)
        # det_gen = generate_detection_results(model, generated_image_paths, False, save_vis)

        # run YOLO tracking and heatmap on all images
        tracks_orig = generate_track_and_heatmap_results(model, original_image_paths, True, save_vis)
        tracks_gen = generate_track_and_heatmap_results(model, generated_image_paths, False, save_vis)

        # Compute undetected objects and related metrics
        count_orig_all, count_undet_all, count_undet_critical, count_undet_noncritial = compute_undetected_objects(tracks_orig, tracks_gen, debug)
        false_negative_rate = count_undet_all / count_orig_all
        false_negative_rate_critical = count_undet_critical / count_orig_all
        false_negative_rate_noncritical = count_undet_noncritial / count_orig_all

        # Compute criticality (area) change and related metrics
        area_changes_orig = compute_area_changes(tracks_orig, segments_length, model.names)
        area_changes_gen = compute_area_changes(tracks_gen, segments_length, model.names)
        res_area_change_orig, res_area_change_gen, res_area_change_comparison = compute_area_change_criticality(area_changes_orig, area_changes_gen, debug)

        # TODO: run 3D object deetctor/tracker from monocamera data (in case compute HW, THW, TTC in a simpler way!)

        # Save results
        results[scene] = {
            'scene': scene,
            'undetected_objects': {
                'count_orig_all': count_orig_all,
                'count_undet_all': count_undet_all,
                'count_undet_critical': count_undet_critical,
                'count_undet_noncritial': count_undet_noncritial,
                'false_negative_rate': false_negative_rate,
                'false_negative_rate_critical': false_negative_rate_critical,
                'false_negative_rate_noncritical': false_negative_rate_noncritical
            },
            'area_changes': {
                'original': res_area_change_orig,
                'generated': res_area_change_gen,
                'comparison': res_area_change_comparison
            }
        }

        # Save join video
        if save_vis:
            print(f"Creating videos for scene: {scene}")
            create_videos_from_images(scene, fps)

        # Plot of safety metrics for each scene
        plot_safety_metrics_for_scene(results[scene])

    return results


# Main function
def main():
    parser = argparse.ArgumentParser(description='Generate front view images from nuScenes dataset')
    parser.add_argument('--original-data-path', type=str, default="/mnt/d/nuscenes/scenes_frames/CAM_FRONT_adj_fov", help='Path to original data directory')
    parser.add_argument('--generated-data-path', type=str, default="/mnt/d/nuscenes/scenes_frames/CAM_FRONT_adj_fov", help='Path to generated data directory')
    parser.add_argument('--segments-length', type=int, default=8, help='Number of frames per video segment')
    parser.add_argument('--fps', type=int, default=12, help='Frames per second for the output video')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the evaluation on')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--save_vis', action='store_true', help='Save visualizations')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugging with debugpy')
    args = parser.parse_args()

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    # load yolo
    model = load_YOLO()

    # calculate safety metrics
    metrics = calculate_safety_for_scenes(args.original_data_path, args.generated_data_path, args.segments_length, args.fps, model, debug=args.debugpy, save_vis=args.save_vis)

    # Save metrics to json file 
    with open('evaluation/safety_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
