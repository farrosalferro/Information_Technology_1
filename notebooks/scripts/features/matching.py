import h5py
import kornia.feature as KF
import numpy as np
import torch
from tqdm import tqdm
import cv2

from ..utils.image import get_image_pairs_exhaustive


def get_image_pairs_shortlist_dino(
    fnames: list[str],
    dino_path: str,
    sim_th: float = 0.6,  # should be strict
    min_pairs: int = 20,
    exhaustive_if_less: int = 20,
    device: torch.device = None,
) -> list[tuple[int, int]]:
    """Get a shortlist of image pairs based on global descriptor similarity (using DINO).

    Args:
        fnames: List of image file paths.
        processor_path: Path to the DINO image processor model.
        model_path: Path to the DINO model.
        sim_th: Similarity threshold for pairing images.
        min_pairs: Minimum number of pairs per image.
        exhaustive_if_less: Use exhaustive matching if image count is less than this value.
        device: Computation device. Default is CPU.

    Returns:
        list[tuple[int, int]]: List of index pairs for matching.
    """
    # Import extract_global_descriptor_dino inside the function to avoid circular imports
    from .extraction import extract_global_descriptor_dino
    
    if device is None:
        device = torch.device("cpu")

    num_imgs = len(fnames)
    # If we have few images, just do exhaustive matching (all pairs)
    if num_imgs <= exhaustive_if_less:
        return get_image_pairs_exhaustive(fnames)

    # Extract global descriptors for all images
    descs = extract_global_descriptor_dino(fnames, dino_path, device=device)
    # Calculate pairwise distances between all descriptors
    dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
    # Find pairs with distance below threshold
    mask = dm <= sim_th
    total = 0
    matching_list = []
    ar = np.arange(num_imgs)

    # For each image, find its matching candidates
    for st_idx in range(num_imgs - 1):
        # Get indices of images that are similar enough (below threshold)
        mask_idx = mask[st_idx]
        to_match = ar[mask_idx]
        # If not enough candidates found, take the top min_pairs by distance
        if len(to_match) < min_pairs:
            to_match = np.argsort(dm[st_idx])[:min_pairs]
        # Add valid pairs to the matching list
        for idx in to_match:
            # Skip self-matches
            if st_idx == idx:
                continue
            # Additional distance check (1000 acts as infinity)
            if dm[st_idx, idx] < 1000:
                # Create the pair and make sure smaller index is first
                matching_list.append(tuple(sorted((st_idx, idx.item()))))
                total += 1
    # Sort and deduplicate pairs
    matching_list = sorted(list(set(matching_list)))
    return matching_list


def match_keypoint_lightglue(
    img_fnames: list[str],
    index_pairs: list[tuple[int, int]],
    feature_dir: str = ".featureout",
    device: torch.device = None,
    min_matches: int = 15,
    verbose: bool = True,
) -> None:
    """Match image pairs using LightGlue and save matches to HDF5 file.

    Args:
        img_fnames: List of image file paths.
        index_pairs: List of image index pairs to match.
        feature_dir: Directory containing precomputed features and to save matches.
        device: Computation device. Default is CPU.
        min_matches: Minimum number of matches required to keep a pair.
        verbose: Whether to print matching information.

    Returns:
        None: Matches are saved to an HDF5 file in the specified directory.
    """
    if device is None:
        device = torch.device("cpu")

    # Initialize LightGlue matcher with ALIKED configuration
    # Disable confidence thresholds for matching, and use mixed precision if on CUDA
    lg_matcher = (
        KF.LightGlueMatcher(
            "aliked",
            {
                "width_confidence": -1,
                "depth_confidence": -1,
                "mp": True if "cuda" in str(device) else False,
            },
        )
        .eval()
        .to(device)
    )

    # Open keypoints and descriptors for reading, and create a file for matches
    with (
        h5py.File(f"{feature_dir}/keypoints.h5", mode="r") as f_kp,
        h5py.File(f"{feature_dir}/descriptors.h5", mode="r") as f_desc,
        h5py.File(f"{feature_dir}/matches.h5", mode="w") as f_match,
    ):
        # Process each pair of images to match
        for pair_idx in tqdm(index_pairs):
            # Get the indices and filenames for this pair
            idx1, idx2 = pair_idx
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            # Extract the base filenames as keys for the HDF5 files
            key1, key2 = fname1.split("/")[-1], fname2.split("/")[-1]

            # Load the keypoints and descriptors for both images
            kp1 = torch.from_numpy(f_kp[key1][...]).to(device)
            kp2 = torch.from_numpy(f_kp[key2][...]).to(device)
            desc1 = torch.from_numpy(f_desc[key1][...]).to(device)
            desc2 = torch.from_numpy(f_desc[key2][...]).to(device)

            # Perform matching using LightGlue
            with torch.inference_mode():
                # Convert keypoints to Local Affine Frames (LAF) format for LightGlue
                dists, idxs = lg_matcher(
                    desc1,
                    desc2,
                    KF.laf_from_center_scale_ori(kp1[None]),
                    KF.laf_from_center_scale_ori(kp2[None]),
                )
            # Skip if no matches found
            if len(idxs) == 0:
                continue

            # Count matches and print if verbose mode is on
            n_matches = len(idxs)
            if verbose:
                print(f"{key1}-{key2}: {n_matches} matches")

            # Create a group for the first image in the HDF5 file
            group = f_match.require_group(key1)
            # Only store matches if they meet the minimum threshold
            if n_matches >= min_matches:
                group.create_dataset(key2, data=idxs.detach().cpu().numpy().reshape(-1, 2))
    return

def get_matcher(matcher_type='FLANN', extractor_type='SIFT'):
    """Creates the specified OpenCV feature matcher."""
    if matcher_type == 'BF':
        # Use appropriate norm type based on descriptor
        if extractor_type in ['SIFT', 'SURF']: # SURF is patented, usually avoid
             norm_type = cv2.NORM_L2
        elif extractor_type in ['ORB', 'BRISK', 'AKAZE']: # Binary descriptors
             norm_type = cv2.NORM_HAMMING
        else: # Default assumption
             norm_type = cv2.NORM_L2
        return cv2.BFMatcher(norm_type, crossCheck=False) # Use crossCheck=False for knnMatch

    elif matcher_type == 'FLANN':
        # FLANN parameters depend on descriptor type
        if extractor_type in ['SIFT', 'SURF']:
            # FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50) # or pass empty dictionary
        elif extractor_type in ['ORB', 'BRISK', 'AKAZE']:
            # FLANN_INDEX_LSH = 6
            # Parameters are tuned for ORB, may need adjustment for others
            index_params= dict(algorithm = 6,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) # 2
            search_params = dict(checks=50) # or pass empty dict
        else:
             raise ValueError(f"FLANN parameters not defined for extractor type: {extractor_type}")
        return cv2.FlannBasedMatcher(index_params, search_params)
    else:
        raise ValueError(f"Unsupported matcher type: {matcher_type}")


def match_and_verify(kps1, descs1, kps2, descs2, matcher, min_inlier_matches_initial=15, lowe_ratio_test_threshold=0.8, ransac_threshold=1.5):
    """Matches features and verifies geometry using Fundamental Matrix RANSAC."""
    if descs1 is None or descs2 is None or len(kps1) < min_inlier_matches_initial or len(kps2) < min_inlier_matches_initial :
        return None, 0 # Not enough keypoints

    # Perform k-Nearest Neighbor matching
    matches = matcher.knnMatch(descs1, descs2, k=2)

    # Filter matches using Lowe's ratio test
    good_matches = []
    try:
        for m, n in matches:
            if m.distance < lowe_ratio_test_threshold * n.distance:
                good_matches.append(m)
    except ValueError:
         # Handle cases where k=1 match is returned (e.g., if descs2 has only 1 feature)
         # print("Warning: knnMatch did not return pairs, possibly too few features in one image.")
         return None, 0


    if len(good_matches) < min_inlier_matches_initial:
        # print(f"Insufficient good matches after ratio test: {len(good_matches)}")
        return None, 0 # Not enough good matches

    # Extract locations of good matches
    pts1 = np.float32([kps1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find Fundamental Matrix using RANSAC
    try:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransac_threshold, confidence=0.99)
    except cv2.error as e:
        # Can happen if points are degenerate (e.g., collinear)
        # print(f"cv2.findFundamentalMat error: {e}")
        return None, 0


    if F is None or mask is None:
        # print("Fundamental matrix estimation failed.")
        return None, 0

    # Count inliers
    num_inliers = int(mask.sum())

    if num_inliers < min_inlier_matches_initial:
        # print(f"Insufficient inliers after RANSAC: {num_inliers}")
        return None, 0

    # Keep only inlier matches
    inlier_matches = [m for i, m in enumerate(good_matches) if mask[i][0] == 1]

    return inlier_matches, num_inliers