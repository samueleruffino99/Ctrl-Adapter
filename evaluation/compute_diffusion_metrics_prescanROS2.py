import torch
import torch.utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
import PIL.Image as Image
from scipy.linalg import sqrtm
from scipy import linalg
import argparse
from tqdm import tqdm
import requests
import hashlib
import glob
import uuid
import io
import html
import re
import scipy

# set seed
torch.manual_seed(0)
np.random.seed(0)


class FIDDataset(torch.utils.data.Dataset):
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
    

class FVDVideoDataset(torch.utils.data.Dataset):
    """Dataset class for FVD calculation."""
    def __init__(self, image_paths, transform=None, segments_length=16):
        self.image_paths = image_paths
        self.transform = transform
        self.segments_length = segments_length
        # create video paths based on segments_length
        self.video_paths = [image_paths[i:i+self.segments_length] for i in range(0, len(image_paths), self.segments_length)]
        self.video_paths = [self.video_paths[0]]
    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_frames = [Image.open(f).convert('RGB') for f in self.video_paths[idx]]
        if self.transform:
            video_frames = [self.transform(frame) for frame in video_frames]
        video_frames = torch.stack(video_frames)
        video_frames = video_frames.permute(1, 0, 2, 3)
        return video_frames


def load_inception_v3():
    """Load the Inception V3 model from PyTorch Hub and remove the classification layer."""
    model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    # Remove the classification layer, 2048-dim feature vector is output
    model.fc = torch.nn.Identity()  
    model.eval()
    return model


def open_url(url, num_attempts=10, verbose=False, cache_dir=None):
    """Open a URL, with multiple attempts and caching."""
    assert num_attempts >=1

    if cache_dir is None:
        cache_dir = './loaded_models'
    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
    if len(cache_files) == 1:
        f_name = cache_files[0]
        return open(f_name, 'rb')
    
    with requests.Session() as session:
        if verbose:
            print("Downloading ", url, flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")
                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise Exception("Interupted")
            except:
                if not attempts_left:
                    if verbose:
                        print("failed!")
                    raise
                if verbose:
                    print('.')
        
    safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
    cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
    temp_file = os.path.join(cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name)
    os.makedirs(cache_dir, exist_ok=True)
    with open(temp_file, 'wb') as f:
        f.write(url_data)
    os.replace(temp_file, cache_file)

    return io.BytesIO(url_data)


# def load_i3d(device='cuda'):
#     """Load the I3D model from PyTorch Hub."""
#     # model = torch.hub.load('kenshohara/video-classification-3d-cnn-pytorch', 'i3d_inception', pretrained=True)
#     # URL from StyleGAN-V repository!
#     _feature_detector_cache = dict()
#     detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
#     detector_kwargs = dict(rescale=True, resize=True, return_features=True) # Return raw features before the softmax layer.
#     key = (detector_url, device)
#     if key not in _feature_detector_cache:
#         with open_url(detector_url, verbose=True) as f:
#             _feature_detector_cache[key] = torch.jit.load(f).eval()
#     return _feature_detector_cache[key]

def load_i3d(device='cuda'):
    """Load the I3D model from PyTorch Hub."""
    detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    with open_url(detector_url, verbose=False) as f:
        detector = torch.jit.load(f).eval().to(device)
    return detector


def get_inception_activations(image_paths, model, batch_size=32, num_workers=1, device='cuda'):
    """Get model activations for a batch of images.
    Args:
        image_paths (list): List of image paths.
        model (torch.nn.Module): Model to extract features.
        batch_size (int): Batch size for feature extraction.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        ndarray: Activations for the input images."""
    model.eval()
    if batch_size > len(image_paths):
        batch_size = len(image_paths)
    dataset = FIDDataset(image_paths, transform=transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    features = []
    for batch in tqdm(dataloader, desc=f"Inception feature extraction"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)
            features.append(pred)
    features = torch.cat(features, dim=0)
    return features.cpu().numpy()


def get_i3d_activations(frames, model, batch_size=32, segments_length=16, num_workers=1, device='cuda'):
    """Get model activations for a batch of videos.
    Args:
        frames (list): List of video frames paths.
        model (torch.nn.Module): Model to extract features.
        batch_size (int): Batch size for feature extraction.
        segments_length (int): Number of frames per video segment.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        ndarray: Activations for the input videos."""
    detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
    model.eval()
    if batch_size > len(frames):
        batch_size = len(frames)
    dataset = FVDVideoDataset(frames, transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]), segments_length=segments_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    all_features = []
    for batch in tqdm(dataloader, desc=f"I3D feature extraction"):
        batch = batch.to(device)
        with torch.no_grad():
            features = model(batch, **detector_kwargs)
            features = features.detach().cpu().numpy()
            all_features.append(features)
    all_features = np.concatenate(all_features, axis=0)
    return all_features
    

def compute_statistics(features):
    """Calculate mean and covariance of features for FID calculation."""
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Args:
        mu1 (ndarray): Mean vector of the first Gaussian.
        sigma1 (ndarray): Covariance matrix of the first Gaussian.
        mu2 (ndarray): Mean vector of the second Gaussian.
        sigma2 (ndarray): Covariance matrix of the second Gaussian.
        eps (float): Small value to avoid numerical issues.
    Returns:
        float: Frechet distance between the two Gaussians."""

    # Ensure numpy arrays
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    # Check dimensions
    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    return fid
    
def compute_fid(original_image_paths, generated_image_paths, inception_model, batch_size=32, num_workers=1, device='cuda'):
    """Calculate FID between original and generated images.
    Args:
        original_image_paths (list): List of paths to original images.
        generated_image_paths (list): List of paths to generated images.
        inception_model (torch.nn.Module): Inception model for feature extraction.
        batch_size (int): Batch size for feature extraction.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        float: FID between the two sets of images."""
    inception_model.to(device)
    original_features = get_inception_activations(original_image_paths, inception_model, batch_size=batch_size, num_workers=num_workers, device=device)
    generated_features = get_inception_activations(generated_image_paths, inception_model, batch_size=batch_size, num_workers=num_workers, device=device)
    mu1, sigma1 = compute_statistics(original_features)
    mu2, sigma2 = compute_statistics(generated_features)
    fid = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


def compute_ssims(original_image_paths, generated_image_paths, batch_size=32, num_workers=1, device='cuda'):
    """Calculate SSIM for each pair of original and generated images.
    Args:
        original_image_paths (list): List of paths to original images.
        generated_image_paths (list): List of paths to generated images.
        batch_size (int): Batch size for feature extraction.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        list: List of SSIM values for each pair of images."""
    original_images = [Image.open(p) for p in original_image_paths]
    generated_images = [Image.open(p) for p in generated_image_paths]
    ssim_values = []
    for original_img, generated_img in zip(original_images, generated_images):
        original_array = np.array(original_img)
        generated_array = np.array(generated_img)
        if original_array.shape == generated_array.shape:
            s = ssim(original_array, generated_array, multichannel=True, channel_axis=2)
            ssim_values.append(s)
    avg_ssim = np.mean(ssim_values)
    return avg_ssim


def compute_fvd(original_images, generated_images, i3d_model, batch_size=32, segments_length=16, num_workers=1, device='cuda'):
    """Calculate FVD between original and generated videos.
    Args:
        original_images (list): List of original images.
        generated_images (list): List of generated images.
        i3d_model (torch.nn.Module): I3D model for feature extraction.
        batch_size (int): Batch size for feature extraction.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        float: FVD between the two sets of videos."""
    i3d_model.to(device)
    original_features = get_i3d_activations(original_images, i3d_model, batch_size=batch_size, segments_length=segments_length, num_workers=num_workers, device=device)
    generated_features = get_i3d_activations(generated_images, i3d_model, batch_size=batch_size, segments_length=segments_length, num_workers=num_workers, device=device)
    # Compute statistics
    mu1, sigma1 = compute_statistics(original_features)
    mu2, sigma2 = compute_statistics(generated_features)
    fvd = compute_frechet_distance(mu1, sigma1, mu2, sigma2)
    return float(fvd)


def calculate_metrics_for_scenes(original_dir, generated_dir, inception_model, i3d_model, batch_size_inception=32, batch_size_i3d=1, segments_length=16, num_workers=1, device='cuda'):
    """Calculate FID, FVD, and SSIM for all scenes and organize in a dictionary.
    Args:
        original_dir (str): Path to the original data directory.
        generated_dir (str): Path to the generated data directory.
        inception_model (torch.nn.Module): Inception model for feature extraction.
        batch_size_inception (int): Batch size for feature extraction with Inception model.
        batch_size_i3d (int): Batch size for feature extraction with I3D model.
        num_workers (int): Number of workers for data loading.
        device (str): Device to run the evaluation on.
    Returns:
        dict: Dictionary containing FID, FVD, and SSIM for each scene."""
    scene_folders = sorted(os.listdir(generated_dir))
    results = {}

    for scene in scene_folders:
        print(f"Processing scene: {scene}")
        results[scene] = {'FID': None, 'FVD': None, 'SSIM': []}
        original_scene_path = os.path.join(original_dir, scene)
        generated_scene_path = os.path.join(generated_dir, scene)

        original_image_paths = sorted([os.path.join(original_scene_path, f) for f in os.listdir(original_scene_path)])
        generated_image_paths = sorted([os.path.join(generated_scene_path, f) for f in os.listdir(generated_scene_path)])
        n_gen_frames = len(generated_image_paths)
        original_image_paths = original_image_paths[:n_gen_frames]
        generated_image_paths = generated_image_paths[:n_gen_frames]

        # Calculate FID
        results[scene]['FID'] = compute_fid(original_image_paths, generated_image_paths, inception_model, batch_size=batch_size_inception, num_workers=num_workers, device=device)

        # Calculate SSIM for each pair of original and generated images
        results[scene]['SSIM'] = compute_ssims(original_image_paths, generated_image_paths, batch_size=batch_size_inception, num_workers=num_workers, device=device)

        # Placeholder for FVD calculation
        results[scene]['FVD'] = compute_fvd(original_image_paths, generated_image_paths, i3d_model, batch_size=batch_size_i3d, segments_length=segments_length, num_workers=num_workers, device=device)

    return results

# Main function
def main():
    parser = argparse.ArgumentParser(description='Generate front view images from nuScenes dataset')
    parser.add_argument('--original-data-path', type=str, default="/mnt/d/nuscenes/scenes_frames/CAM_FRONT_adj_fov", help='Path to original data directory')
    parser.add_argument('--generated-data-path', type=str, default="/mnt/d/nuscenes/prescanros2_diffused_video", help='Path to generated data directory')
    parser.add_argument('--segments-length', type=int, default=16, help='Number of frames per video segment')
    parser.add_argument('--device', type=str, default="cuda", help='Device to run the evaluation on')
    parser.add_argument('--num-workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--batch-size-inception', type=int, default=32, help='Batch size for data loading')
    parser.add_argument('--batch-size-i3d', type=int, default=1, help='Batch size for data loading')
    parser.add_argument('--debugpy', action='store_true', help='Enable debugging with debugpy')
    args = parser.parse_args()

    if args.debugpy:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()

    if args.device is None:
        device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers
    
    # Load the models for FID and FVD using PyTorch Hub
    inception_model = load_inception_v3()
    i3d_model = load_i3d()

    metrics = calculate_metrics_for_scenes(args.original_data_path, args.generated_data_path, inception_model, i3d_model, batch_size_inception=args.batch_size_inception, batch_size_i3d=args.batch_size_i3d , segments_length=args.segments_length , num_workers=num_workers, device=device)
    for scene, data in metrics.items():
        print(f"Scene: {scene}, FID: {data['FID']}, Average SSIM: {np.mean(data['SSIM'])}")
    # compute average among all scenes
    fid_values = [data['FID'] for data in metrics.values()]
    ssim_values = [np.mean(data['SSIM']) for data in metrics.values()]
    print(f"Average FID: {np.mean(fid_values)}, Average SSIM: {np.mean(ssim_values)}, Average FVD: {data['FVD']}")

if __name__ == "__main__":
    main()
