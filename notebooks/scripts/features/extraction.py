import os
from typing import Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from lightglue import ALIKED
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel
from umap import UMAP
import cv2
from pathlib import Path

from ..utils.image import load_torch_image


def extract_global_descriptor_dino(
    fnames: list[str],
    dino_path: str,
    device: torch.device = None,
) -> torch.Tensor:
    """Extract global descriptors for a list of images using DINO.

    Args:
        fnames: List of image file paths.
        processor_path: Path to the DINO image processor model.
        model_path: Path to the DINO model.
        device: Computation device. Default is CPU.

    Returns:
        torch.Tensor: Matrix of global descriptors with shape [N, D] where N is the number of images.
    """
    if device is None:
        device = torch.device("cpu")

    # Load the DINO model and processor from pre-trained weights
    processor = AutoImageProcessor.from_pretrained(dino_path)
    model = AutoModel.from_pretrained(dino_path)
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []

    # Process each image to extract its global descriptor
    for _, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        # Load image as tensor
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            # Process image with DINO without rescaling (preserves original pixel values)
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            # Create global descriptor by taking the max over spatial dimensions of hidden states
            # Skip the CLS token (index 0) and L2-normalize the result
            dino_mac = F.normalize(outputs.last_hidden_state[:, 1:].max(dim=1)[0], dim=1, p=2)
        # Store descriptor and free GPU memory by moving to CPU
        global_descs_dinov2.append(dino_mac.detach().cpu())
    # Concatenate all descriptors into a single tensor
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2


def extract_cls_descriptor_dino(
    fnames: list[str],
    dino_path: str,
    device: torch.device = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Extract global descriptors for a list of images using DINO. USE CLS TOKENS.

    Args:
        fnames: List of image file paths.
        processor_path: Path to the DINO image processor model.
        model_path: Path to the DINO model.
        device: Computation device. Default is CPU.

    Returns:
        torch.Tensor: Matrix of global descriptors with shape [N, D] where N is the number of images.
    """
    if device is None:
        device = torch.device("cpu")

    # Load the DINO model and processor from pre-trained weights
    processor = AutoImageProcessor.from_pretrained(dino_path)
    model = AutoModel.from_pretrained(dino_path)
    model = model.eval()
    model = model.to(device)
    global_descs_dinov2 = []

    # Process each image to extract its global descriptor
    for _, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
        # Load image as tensor
        timg = load_torch_image(img_fname_full)
        with torch.inference_mode():
            # Process image with DINO without rescaling (preserves original pixel values)
            inputs = processor(images=timg, return_tensors="pt", do_rescale=False).to(device)
            outputs = model(**inputs)
            # Create global descriptor by taking the CLS token (index 0)
            if normalize:
                dino_mac = F.normalize(outputs.last_hidden_state[:, 0, :], dim=-1, p=2)
        # Store descriptor and free GPU memory by moving to CPU
        global_descs_dinov2.append(dino_mac.detach().cpu())
    # Concatenate all descriptors into a single tensor
    global_descs_dinov2 = torch.cat(global_descs_dinov2, dim=0)
    return global_descs_dinov2


def detect_keypoint_aliked(
    img_fnames: list[str],
    feature_dir: str = ".featureout",
    num_features: int = 4096,
    resize_to: int = 1024,
    device: torch.device = None,
) -> None:
    """Extract ALIKED features for a list of images and save to HDF5 files.

    Args:
        img_fnames: List of image file paths.
        feature_dir: Directory to save the features.
        num_features: Maximum number of keypoints to detect per image.
        resize_to: Size to resize images to before feature extraction.
        device: Computation device. Default is CPU.

    Returns:
        None: Features are saved to HDF5 files in the specified directory.
    """
    # ALIKED works better with float32 for numerical stability
    dtype = torch.float32  # ALIKED has issues with float16
    if device is None:
        device = torch.device("cpu")
    # Initialize the ALIKED keypoint detector and descriptor extractor
    extractor = (
        ALIKED(max_num_keypoints=num_features, detection_threshold=0.01, resize=resize_to)
        .eval()
        .to(device, dtype)
    )

    # Create output directory if it doesn't exist
    if not os.path.isdir(feature_dir):
        os.makedirs(feature_dir)

    # Open HDF5 files for storing keypoints and descriptors
    with (
        h5py.File(f"{feature_dir}/keypoints.h5", mode="w") as f_kp,
        h5py.File(f"{feature_dir}/descriptors.h5", mode="w") as f_desc,
    ):
        # Process each image
        for img_path in tqdm(img_fnames):
            # Extract filename to use as key in HDF5 file
            img_fname = img_path.split("/")[-1]
            key = img_fname
            with torch.inference_mode():
                # Load and process the image
                image0 = load_torch_image(img_path, device=device).to(dtype)
                # Extract features (keypoints and descriptors)
                feats0 = extractor.extract(
                    image0
                )  # auto-resize the image, disable with resize=None
                # Reshape and move to CPU for storage
                kpts = feats0["keypoints"].reshape(-1, 2).detach().cpu().numpy()
                descs = feats0["descriptors"].reshape(len(kpts), -1).detach().cpu().numpy()
                # Store in HDF5 files
                f_kp[key] = kpts
                f_desc[key] = descs
    return


def feature_reducer(
    algorithm: str,
    features: np.ndarray,
    n_components: int,
    scaler: Union[None, str] = None,
    **kwargs,
) -> np.ndarray:
    """Reduce the dimensionality of features using the specified algorithm.

    Args:
        algorithm: Dimensionality reduction algorithm ('PCA', 'UMAP', or 'TSNE').
        scaler: Optional scaler to apply ('StandardScaler', 'MinMaxScaler', or None).
        features: Input feature matrix.
        n_components: Number of components to keep.
        scaler: Optional scaler to apply ('StandardScaler', 'MinMaxScaler', or None).
        **kwargs: Additional arguments for the chosen algorithm.

    Returns:
        np.ndarray: Reduced feature matrix.
    """
    if features.ndim != 2:
        raise ValueError(f"Input features should be a 2D array, got shape {features.shape}")

    # Apply scaling if specified
    if scaler == "StandardScaler":
        features = StandardScaler().fit_transform(features)
    elif scaler == "MinMaxScaler":
        features = MinMaxScaler().fit_transform(features)
    elif scaler is None or scaler == "None":
        pass
    else:
        raise ValueError(f"Unsupported scaler: {scaler}")

    # Apply dimensionality reduction
    if algorithm == "PCA":
        reducer = PCA(n_components=n_components, **kwargs)
    elif algorithm == "UMAP":
        reducer = UMAP(n_components=n_components, **kwargs)
    elif (
        algorithm == "TSNE"
    ):  # actually tsne is not a feature reducer algorithm. Read more here (https://datascience.stackexchange.com/questions/96944/why-it-is-recommended-to-use-t-sne-to-reduce-to-2-3-dims-and-not-higher-dim)
        if n_components > 3:
            raise ValueError("TSNE is typically used with 2 or 3 components.")
        reducer = TSNE(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    return reducer.fit_transform(features)


def get_feature_extractor(extractor_type='SIFT', nfeatures=8000):
    """Creates the specified OpenCV feature extractor."""
    if extractor_type == 'SIFT':
        return cv2.SIFT_create(nfeatures=nfeatures)
    elif extractor_type == 'AKAZE':
        # AKAZE needs descriptors of size 61*8 = 488 bits
        return cv2.AKAZE_create()
    elif extractor_type == 'ORB':
        return cv2.ORB_create(nfeatures=nfeatures) # Use same nfeatures param for consistency
    # Add other types like BRISK if needed
    else:
        raise ValueError(f"Unsupported feature extractor type: {extractor_type}")


def extract_features(image_path, extractor):
    """Extracts keypoints and descriptors from an image file."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not read image {image_path}")
        return None, None, (None, None)

    # Check if image dimensions are too small (might cause issues)
    h, w = img.shape
    if h < 20 or w < 20: # Example threshold
        # print(f"Warning: Image {image_path} is very small ({w}x{h}), skipping feature extraction.")
        return None, None, (w,h)


    kps, descs = extractor.detectAndCompute(img, None)

    if kps is None or descs is None or len(kps) == 0:
         return None, None, (w,h)

    return kps, descs, (w, h) 


def load_and_extract_features_dataset(dataset_id, dataset_base_path, extractor):
    """Loads all images for a dataset and extracts features."""
    features = {}
    image_dims = {}
    dataset_path = Path(dataset_base_path) / dataset_id
    image_files = list(dataset_path.glob('*.png')) + list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.jpeg'))

    # Handle potential 'outliers' subdirectory - check if needed based on actual data structure
    outlier_path = Path(dataset_path) / 'outliers'
    if outlier_path.is_dir():
        print(f"Including images from outliers subdirectory for {dataset_id}")
        image_files.extend(list(outlier_path.glob('*.png')) + list(outlier_path.glob('*.jpg')) + list(outlier_path.glob('*.jpeg')))


    print(f"Extracting features for {len(image_files)} images in dataset {dataset_id}...")
    for img_path in tqdm(image_files, desc=f"Features {dataset_id}"):
        image_id = img_path.name # Use filename as unique ID within dataset
        kps, descs, dims = extract_features(img_path, extractor)
        if kps is not None and descs is not None:
            features[image_id] = (kps, descs)
            image_dims[image_id] = dims
        else:
             # Store dims even if features failed, might be needed for K matrix
             if dims[0] is not None:
                 image_dims[image_id] = dims

    return features, image_dims