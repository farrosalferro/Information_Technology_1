# This is the IMC 3D error metric code

import csv
import math
import warnings

import numpy as np

# Small epsilon value to avoid numerical issues in calculations
# Used primarily in quaternion operations and matrix calculations
_EPS = np.finfo(float).eps * 4.0


def read_csv(filename: str, header: bool = True, print_header: bool = False) -> dict:
    """Read a CSV file containing camera pose data.

    Args:
        filename: Path to the CSV file.
        header: Whether the CSV file has a header row.
        print_header: Whether to print the header row.

    Returns:
        A nested dictionary with structure {dataset: {scene: {image: {R, t, c}}}}
        where R is the rotation matrix, t is the translation vector, and c is the camera center.
    """
    data = {}
    label_idx = {}

    with open(filename, newline="\n") as csvfile:
        csv_lines = csv.reader(csvfile, delimiter=",")
        for row in csv_lines:
            if header:
                # Process header row to get column indices
                header = False  # Set to false so we only process header once
                for i, name in enumerate(row):
                    label_idx[name] = i  # Map column names to their indices
                if print_header:
                    print(f"Skipping header for file {filename}: {row}")
                continue

            # Extract data from the current row
            dataset = row[label_idx["dataset"]]
            scene = row[label_idx["scene"]]
            image = row[label_idx["image"]]

            # Parse rotation matrix from semicolon-separated string and reshape to 3x3
            R = np.array(
                [float(x) for x in (row[label_idx["rotation_matrix"]].split(";"))]
            ).reshape(3, 3)

            # Parse translation vector from semicolon-separated string
            t = np.array(
                [float(x) for x in (row[label_idx["translation_vector"]].split(";"))]
            ).reshape(3)

            # Calculate camera center: c = -R^T * t
            # This is the position of the camera in world coordinates
            c = -R.T @ t

            # Build the nested dictionary structure
            if dataset not in data:
                data[dataset] = {}
            if scene not in data[dataset]:
                data[dataset][scene] = {}
            data[dataset][scene][image] = {"R": R, "t": t, "c": c}
    return data


def quaternion_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return homogeneous rotation matrix from quaternion.

    Args:
        quaternion: A quaternion represented as a numpy array.

    Returns:
        A 4x4 homogeneous rotation matrix.
    """
    # Make a copy of the quaternion to avoid modifying the input
    q = np.array(quaternion, dtype=np.float64, copy=True)

    # Calculate the squared norm of the quaternion
    n = np.dot(q, q)

    # Handle the case of a very small quaternion (nearly zero)
    if n < _EPS:
        # Return identity matrix for degenerate quaternion
        return np.identity(4)

    # Normalize the quaternion with a specific scaling factor
    q *= math.sqrt(2.0 / n)

    # Compute the outer product of the quaternion with itself
    # This is used to construct the rotation matrix
    q = np.outer(q, q)

    # Construct the 4x4 homogeneous rotation matrix using quaternion components
    # The 3x3 upper-left submatrix represents the rotation
    # The bottom row [0,0,0,1] makes it a homogeneous transformation matrix
    return np.array(
        [
            [
                1.0 - q[2, 2] - q[3, 3],  # R[0,0]
                q[1, 2] - q[3, 0],  # R[0,1]
                q[1, 3] + q[2, 0],  # R[0,2]
                0.0,  # No translation
            ],
            [
                q[1, 2] + q[3, 0],  # R[1,0]
                1.0 - q[1, 1] - q[3, 3],  # R[1,1]
                q[2, 3] - q[1, 0],  # R[1,2]
                0.0,  # No translation
            ],
            [
                q[1, 3] - q[2, 0],  # R[2,0]
                q[2, 3] + q[1, 0],  # R[2,1]
                1.0 - q[1, 1] - q[2, 2],  # R[2,2]
                0.0,  # No translation
            ],
            [0.0, 0.0, 0.0, 1.0],  # Homogeneous row
        ]
    )


def mAA_on_cameras(
    err: np.ndarray, thresholds: list, n: int, skip_top_thresholds: int, to_dec: int = 3
) -> float:
    """Calculate the mean Average Accuracy (mAA) for camera registration.

    mAA is the mean of mAA_i, where for each threshold th_i in <thresholds>, excluding the first <skip_top_thresholds values>,
    mAA_i = max(0, sum(err_i < th_i) - <to_dec>) / (n - <to_dec>)
    where <n> is the number of ground-truth cameras and err_i is the camera registration error for the best
    registration corresponding to threshold th_i.

    Args:
        err: Array of camera registration errors.
        thresholds: List of thresholds for evaluation.
        n: Number of ground-truth cameras.
        skip_top_thresholds: Number of top thresholds to skip.
        to_dec: Deduction value for the calculation.

    Returns:
        The mean Average Accuracy score.
    """
    # Create a boolean mask where True indicates errors below the threshold
    # We skip the first skip_top_thresholds values and expand the thresholds array
    # to match the shape of the error array for broadcasting
    aux = err[:, skip_top_thresholds:] < np.expand_dims(
        np.asarray(thresholds[skip_top_thresholds:]), axis=0
    )

    # Count how many cameras are below each threshold, subtract the deduction value,
    # and ensure the result is not negative
    numerator = np.sum(np.maximum(np.sum(aux, axis=0) - to_dec, 0))

    # Calculate the final mAA score by dividing by the total possible score
    # If numerator is 0, return 0 to avoid division by zero
    return (
        0 if numerator == 0 else numerator / (len(thresholds[skip_top_thresholds:]) * (n - to_dec))
    )


def mAA_on_cameras_per_th(err: np.ndarray, thresholds: list, n: int, to_dec: int = 3) -> np.ndarray:
    """Calculate the mean Average Accuracy (mAA) per threshold.

    Similar to mAA_on_cameras, but returns the score for each threshold separately.
    To be used in score_all_ext with per_th=True.

    Args:
        err: Array of camera registration errors.
        thresholds: List of thresholds for evaluation.
        n: Number of ground-truth cameras.
        to_dec: Deduction value for the calculation.

    Returns:
        Array of mAA scores, one for each threshold.
    """
    # Create a boolean mask where True indicates errors below the threshold
    # Expand the thresholds array to match the shape of the error array for broadcasting
    aux = err < np.expand_dims(np.asarray(thresholds), axis=0)

    # For each threshold, count how many cameras are below it, subtract the deduction value,
    # ensure the result is not negative, and normalize by (n - to_dec)
    # This returns an array with one mAA score per threshold
    return np.maximum(np.sum(aux, axis=0) - to_dec, 0) / (n - to_dec)


def check_data(gt_data: dict, user_data: dict, print_error: bool = False) -> bool:
    """Check if the ground truth and submission data are correct.

    Verifies that:
    - Images in different scenes in the same dataset cannot have the same name in gt_data
    - There must be exactly an entry for each dataset, scene, image entry in the gt_data

    Args:
        gt_data: Ground truth data dictionary.
        user_data: User submission data dictionary.
        print_error: Whether to print error messages. ATTENTION: must be disabled when
                    called from score_all_ext to avoid possible data leaks!

    Returns:
        True if the data is valid, False otherwise.
    """

    for dataset in gt_data.keys():
        # Create a dictionary to track all images in this dataset
        aux = {}

        # First check: ensure no duplicate image names across scenes in ground truth
        for scene in gt_data[dataset].keys():
            for image in gt_data[dataset][scene].keys():
                if image in aux:
                    # Found a duplicate image name in the ground truth
                    if print_error:
                        warnings.warn(
                            f"image {image} found duplicated in the GT dataset {dataset}",
                            stacklevel=1,
                        )
                    return False
                else:
                    # Mark this image as seen
                    aux[image] = 1

        # Second check: ensure the dataset exists in user submission
        if dataset not in user_data.keys():
            if print_error:
                warnings.warn(f"dataset {dataset} not found in submission", stacklevel=1)
            return False

        # Third check: ensure all images in user submission belong to ground truth
        for scene in user_data[dataset].keys():
            for image in user_data[dataset][scene].keys():
                if image not in aux:
                    # Found an image in submission that doesn't exist in ground truth
                    if print_error:
                        warnings.warn(
                            f"image {image} does not belong to the GT dataset {dataset}",
                            stacklevel=1,
                        )
                    return False
                else:
                    # Remove the image from aux to track which ones have been processed
                    aux.pop(image)

        # Fourth check: ensure all ground truth images are in the submission
        if len(aux) > 0:
            # Some images from ground truth are missing in the submission
            if print_error:
                warnings.warn(f"submission dataset {dataset} missing some GT images", stacklevel=1)
            return False

    # All checks passed
    return True


def register_by_Horn(
    ev_coord: np.ndarray,
    gt_coord: np.ndarray,
    ransac_threshold: np.ndarray,
    inl_cf: float,
    strict_cf: float,
) -> dict:
    """Find the best similarity transforms that register 3D points using Horn's method.

    This function implements a RANSAC-like approach to find the best similarity transform
    that aligns evaluation camera centers with ground truth camera centers. It's a key part
    of the evaluation process for the IMC competition.

    Returns the best similarity transforms T that registers 3D points pt_ev in <ev_coord> to
    the corresponding ones pt_gt in <gt_coord> according to a RANSAC-like approach for each
    threshold value th in <ransac_threshold>.

    Given th, each triplet of 3D correspondences is examined if not already present as strict inlier,
    a correspondence is a strict inlier if <strict_cf> * err_best < th, where err_best is the registration
    error for the best model so far.
    The minimal model given by the triplet is then refined using also its inliers if their total is greater
    than <inl_cf> * ninl_best, where ninl_best is th number of inliers for the best model so far. Inliers
    are 3D correspondences (pt_ev, pt_gt) for which the Euclidean distance |pt_gt-T*pt_ev| is less than th.

    Args:
        ev_coord: Coordinates of the evaluation points (camera centers from submission).
        gt_coord: Coordinates of the ground truth points (ground truth camera centers).
        ransac_threshold: Thresholds for RANSAC, used to determine inliers.
        inl_cf: Inlier confidence factor, controls when to refine a model.
        strict_cf: Strict confidence factor, controls when a point is a strict inlier.

    Returns:
        Dictionary containing the best model information including valid cameras, number of inliers,
        errors, triplets used, and transformation matrices.
    """

    # Remove invalid cameras (those with NaN or Inf values)
    # The index of valid cameras is returned for later reference
    idx_cams = np.all(np.isfinite(ev_coord), axis=0)
    ev_coord = ev_coord[:, idx_cams]
    gt_coord = gt_coord[:, idx_cams]

    # Initialization
    n = ev_coord.shape[1]  # Number of valid cameras
    r = ransac_threshold.shape[0]  # Number of thresholds

    # Reshape threshold array for broadcasting and compute squared thresholds
    ransac_threshold = np.expand_dims(ransac_threshold, axis=0)
    ransac_threshold2 = ransac_threshold**2

    # Add homogeneous coordinate (1) to evaluation coordinates for transformation
    ev_coord_1 = np.vstack((ev_coord, np.ones(n)))

    # Initialize arrays to track the best model for each threshold
    max_no_inl = np.zeros((1, r))  # Maximum number of inliers
    best_inl_err = np.full(r, np.inf)  # Best inlier error sum
    best_transf_matrix = np.zeros((r, 4, 4))  # Best transformation matrices
    best_err = np.full((n, r), np.inf)  # Best errors for each camera and threshold
    strict_inl = np.full((n, r), False)  # Strict inlier flags
    triplets_used = np.zeros((3, r))  # Triplets used for best models

    # Run RANSAC on all possible camera triplets
    # This is the core of the algorithm - try all possible triplets of cameras
    # to find the best registration between evaluation and ground truth
    for ii in range(n - 2):
        for jj in range(ii + 1, n - 1):
            for kk in range(jj + 1, n):
                i = [ii, jj, kk]  # Current triplet
                triplets_used_now = np.full((n), False)
                triplets_used_now[i] = True

                # Skip if all cameras in this triplet are already strict inliers
                # for the best current model (optimization)
                if np.all(strict_inl[i]):
                    continue

                # Get transformation T by Horn's method on the triplet camera center correspondences
                transf_matrix = affine_matrix_from_points(
                    ev_coord[:, i], gt_coord[:, i], usesvd=False
                )

                # Apply transformation T to all evaluation camera centers
                rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)

                # Compute squared error for each camera center
                err = np.sum((rotranslated - gt_coord) ** 2, axis=0)

                # Determine inliers for each threshold
                inl = np.expand_dims(err, axis=1) < ransac_threshold2
                no_inl = np.sum(inl, axis=0)  # Count inliers for each threshold

                # Determine which thresholds need refinement
                # A threshold needs refinement if:
                # 1. It has more than 2 inliers
                # 2. It has more inliers than inl_cf * max_no_inl (current best)
                to_ref = np.squeeze(((no_inl > 2) & (no_inl > max_no_inl * inl_cf)), axis=0)

                # Refine models for thresholds that need it
                for q in np.argwhere(to_ref):
                    qq = q[0]  # Current threshold index

                    # Skip if we've already processed this exact set of inliers
                    if np.any(np.all((np.expand_dims(inl[:, qq], axis=1) == inl[:, :qq]), axis=0)):
                        continue

                    # Refine transformation using all inliers for this threshold
                    # This is where we improve the initial estimate from the triplet
                    transf_matrix = affine_matrix_from_points(
                        ev_coord[:, inl[:, qq]], gt_coord[:, inl[:, qq]]
                    )

                    # Apply refined transformation to all evaluation camera centers
                    rotranslated = np.matmul(transf_matrix[:3], ev_coord_1)

                    # Compute errors and count inliers for the refined model
                    err_ref = np.sum((rotranslated - gt_coord) ** 2, axis=0)
                    err_ref_sum = np.sum(err_ref, axis=0)
                    err_ref = np.expand_dims(err_ref, axis=1)
                    inl_ref = err_ref < ransac_threshold2
                    no_inl_ref = np.sum(inl_ref, axis=0)

                    # Determine which thresholds to update with this refined model
                    # Update if:
                    # 1. New model has more inliers, or
                    # 2. Same number of inliers but lower total error
                    to_update = np.squeeze(
                        (no_inl_ref > max_no_inl)
                        | ((no_inl_ref == max_no_inl) & (err_ref_sum < best_inl_err)),
                        axis=0,
                    )

                    # Update best model information for thresholds that improved
                    if np.any(to_update):
                        triplets_used[0, to_update] = ii
                        triplets_used[1, to_update] = jj
                        triplets_used[2, to_update] = kk
                        max_no_inl[:, to_update] = no_inl_ref[to_update]
                        best_err[:, to_update] = np.sqrt(err_ref)  # Store error as distance
                        best_inl_err[to_update] = err_ref_sum

                        # Update strict inlier flags
                        # A point is a strict inlier if its error is less than strict_cf * threshold
                        strict_inl[:, to_update] = (
                            best_err[:, to_update] < strict_cf * ransac_threshold[:, to_update]
                        )

                        # Store the best transformation matrix
                        best_transf_matrix[to_update] = transf_matrix

    # Package all the best model information into a dictionary
    best_model = {
        "valid_cams": idx_cams,  # Indices of valid cameras
        "no_inl": max_no_inl,  # Number of inliers for each threshold
        "err": best_err,  # Error for each camera and threshold
        "triplets_used": triplets_used,  # Triplets used for best models
        "transf_matrix": best_transf_matrix,  # Best transformation matrices
    }
    return best_model


def affine_matrix_from_points(
    v0: np.ndarray, v1: np.ndarray, shear: bool = False, scale: bool = True, usesvd: bool = True
) -> np.ndarray:
    """Calculate affine transform matrix to register two point sets.

    This function finds the optimal transformation matrix that maps points in v0 to
    corresponding points in v1. It's a key component of the Horn's method used in
    the register_by_Horn function for camera registration.

    Args:
        v0: Source points, shape (ndims, -1) array of at least ndims non-homogeneous coordinates.
        v1: Target points, shape (ndims, -1) array of at least ndims non-homogeneous coordinates.
        shear: If False, a similarity transformation matrix is returned.
        scale: If False and shear is False, a rigid/Euclidean transformation matrix is returned.
        usesvd: If True, uses SVD-based algorithm (Kabsch). If False and ndims is 3,
                uses quaternion-based algorithm (Horn).

    Returns:
        Affine transformation matrix that maps v0 to v1.

    Notes:
        By default the algorithm by Hartley and Zissermann [15] is used.
        If usesvd is True, similarity and Euclidean transformation matrices
        are calculated by minimizing the weighted sum of squared deviations
        (RMSD) according to the algorithm by Kabsch [8].
        Otherwise, and if ndims is 3, the quaternion based algorithm by Horn [9]
        is used, which is slower when using this Python implementation.
        The returned matrix performs rotation, translation and uniform scaling
        (if specified).
    """

    # Make copies of input arrays to avoid modifying them
    v0 = np.array(v0, dtype=np.float64, copy=True)
    v1 = np.array(v1, dtype=np.float64, copy=True)

    # Get dimensionality and validate input shapes
    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # Step 1: Move centroids of both point sets to origin
    # This is a standard preprocessing step in point set registration
    t0 = -np.mean(v0, axis=1)  # Translation to move v0 centroid to origin
    M0 = np.identity(ndims + 1)  # Homogeneous transformation matrix for v0
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)  # Apply translation to v0

    t1 = -np.mean(v1, axis=1)  # Translation to move v1 centroid to origin
    M1 = np.identity(ndims + 1)  # Homogeneous transformation matrix for v1
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)  # Apply translation to v1

    # Step 2: Calculate transformation based on the specified method
    if shear:
        # Full affine transformation (allows shearing)
        A = np.concatenate((v0, v1), axis=0)
        u, s, vh = np.linalg.svd(A.T)
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims : 2 * ndims]
        t = np.dot(C, np.linalg.pinv(B))  # Calculate transformation matrix
        t = np.concatenate((t, np.zeros((ndims, 1))), axis=1)
        M = np.vstack((t, ((0.0,) * ndims) + (1.0,)))  # Make homogeneous
    elif usesvd or ndims != 3:
        # Rigid transformation via SVD of covariance matrix (Kabsch algorithm)
        # This finds the optimal rotation between the point sets
        u, s, vh = np.linalg.svd(np.dot(v1, v0.T))  # SVD of covariance matrix

        # Calculate rotation matrix from SVD orthonormal bases
        R = np.dot(u, vh)

        # Ensure we have a proper rotation matrix (determinant = 1)
        if np.linalg.det(R) < 0.0:
            # If determinant is negative, we need to flip one axis
            # This ensures we have a right-handed coordinate system
            R -= np.outer(u[:, ndims - 1], vh[ndims - 1, :] * 2.0)
            s[-1] *= -1.0

        # Create homogeneous transformation matrix with rotation
        M = np.identity(ndims + 1)
        M[:ndims, :ndims] = R
    else:
        # Rigid transformation matrix via quaternion (Horn's method)
        # This is an alternative method for finding the optimal rotation
        # Compute elements of the symmetric 4x4 matrix N
        xx, yy, zz = np.sum(v0 * v1, axis=1)
        xy, yz, zx = np.sum(v0 * np.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = np.sum(v0 * np.roll(v1, -2, axis=0), axis=1)

        # Construct the N matrix used in Horn's quaternion method
        N = [
            [xx + yy + zz, 0.0, 0.0, 0.0],
            [yz - zy, xx - yy - zz, 0.0, 0.0],
            [zx - xz, xy + yx, yy - xx - zz, 0.0],
            [xy - yx, zx + xz, yz + zy, zz - xx - yy],
        ]

        # Find quaternion as eigenvector corresponding to largest eigenvalue
        w, V = np.linalg.eigh(N)
        q = V[:, np.argmax(w)]
        q /= np.linalg.norm(q + _EPS)  # Normalize to unit quaternion

        # Convert quaternion to homogeneous transformation matrix
        M = quaternion_matrix(q)

    # Step 3: Apply scaling if requested (and not using full affine transformation)
    if scale and not shear:
        # Calculate scaling factor as ratio of RMS deviations from centroid
        v0 *= v0  # Square each element for RMS calculation
        v1 *= v1
        # Apply scaling to the rotation part of the matrix
        M[:ndims, :ndims] *= math.sqrt(np.sum(v1) / np.sum(v0))

    # Step 4: Combine transformations to get final result
    # This applies: move v0 to origin, transform, move back according to v1
    M = np.dot(np.linalg.inv(M1), np.dot(M, M0))

    # Normalize the homogeneous coordinate
    M /= M[ndims, ndims]

    return M


def tth_from_csv(csv_file: str) -> tuple:
    """Read thresholds from a CSV file.

    Parses a CSV file containing threshold values for each dataset and scene.
    These thresholds are used in the evaluation process to determine when a camera
    is considered correctly registered.

    Args:
        csv_file: Path to the CSV file containing thresholds.

    Returns:
        A tuple containing:
        - A nested dictionary with structure {dataset: {scene: thresholds}}
        - The number of thresholds

    Raises:
        ValueError: If the number of thresholds varies per scene.
    """

    tth = {}  # Dictionary to store thresholds
    label_idx = {}  # Dictionary to map column names to indices
    n_thresholds = []  # List to track the number of thresholds for each scene

    with open(csv_file, newline="\n") as csvfile:
        csv_lines = csv.reader(csvfile, delimiter=",")
        header = True
        for row in csv_lines:
            if header:
                # Process header row
                header = False
                for i, name in enumerate(row):
                    label_idx[name] = i
                continue
            if not row:
                # Skip empty rows
                continue

            # Extract data from the current row
            dataset = row[label_idx["dataset"]]
            scene = row[label_idx["scene"]]

            # Parse thresholds from semicolon-separated string
            th = np.array([float(x) for x in (row[label_idx["thresholds"]].split(";"))])
            n_thresholds.append(len(th))

            # Build the nested dictionary structure
            if dataset not in tth:
                tth[dataset] = {}
            tth[dataset][scene] = th

    # Ensure all scenes have the same number of thresholds
    if len(set(n_thresholds)) != 1:
        raise ValueError(f"Number of thresholds vary per scene: {list(set(n_thresholds))}")

    return tth, n_thresholds[0]  # Return thresholds dictionary and number of thresholds


def generate_mask_all_public(gt_data: dict) -> dict:
    """Generate a mask marking all images as public.

    This function creates a mask that marks all images in the ground truth data as public.
    It's used when no specific mask file is provided, treating all data as part of the
    public split for evaluation purposes.

    Args:
        gt_data: Ground truth data dictionary with structure {dataset: {scene: {image: data}}}.

    Returns:
        A nested dictionary with the same structure as gt_data, with all values set to True.
    """
    # Create an empty mask dictionary with the same structure as gt_data
    mask = {}

    # Iterate through all datasets, scenes, and images in gt_data
    for dataset in gt_data:
        if dataset not in mask:
            mask[dataset] = {}
        for scene in gt_data[dataset]:
            if scene not in mask[dataset]:
                mask[dataset][scene] = {}
            # Mark each image as public (True)
            for image in gt_data[dataset][scene]:
                mask[dataset][scene][image] = True

    return mask


def fuse_score(mAA_score: float, cluster_score: float, combo_mode: str) -> float:
    """Combine mAA score and clustering score using the specified method.

    This function combines the mean Average Accuracy (mAA) score and the clustering score
    using one of several methods to produce a final evaluation score. The default method
    is the harmonic mean (F1 score), which balances precision (clusterness) and recall (mAA).

    Args:
        mAA_score: The mean Average Accuracy score (recall-like metric).
        cluster_score: The clustering score (precision-like metric).
        combo_mode: The method to combine scores. Options are:
                   "harmonic" (F1 score), "geometric", "arithmetic", "mAA", or "clusterness".

    Returns:
        The combined score as a float between 0 and 1.
    """
    if combo_mode == "harmonic":
        # Harmonic mean (F1 score): 2 * (precision * recall) / (precision + recall)
        if (mAA_score + cluster_score) == 0:
            score = 0  # Avoid division by zero
        else:
            score = 2 * mAA_score * cluster_score / (mAA_score + cluster_score)
    elif combo_mode == "geometric":
        # Geometric mean: sqrt(precision * recall)
        score = (mAA_score * cluster_score) ** 0.5
    elif combo_mode == "arithmetic":
        # Arithmetic mean: (precision + recall) / 2
        # Note: This is not recommended as it can give high scores even if one metric is zero
        score = (mAA_score + cluster_score) * 0.5
    elif combo_mode == "mAA":
        # Use only the mAA score (recall-like metric)
        score = mAA_score
    elif combo_mode == "clusterness":
        # Use only the clustering score (precision-like metric)
        score = cluster_score

    return score


def get_clusterness_score(best_cluster: np.ndarray, best_user_scene_sum: np.ndarray) -> float:
    """Calculate the clusterness score.

    The clusterness score is the ratio of the number of images in both the ground truth and
    user cluster to the total number of images in the user cluster.

    This is essentially a precision metric: how many of the images in the user's cluster
    actually belong to the ground truth scene.

    Args:
        best_cluster: Array containing the number of images in both ground truth and user clusters.
        best_user_scene_sum: Array containing the total number of images in user clusters.

    Returns:
        The clusterness score, a value between 0 and 1.
    """
    # Calculate the total number of images that are in both ground truth and user clusters
    n = np.sum(best_cluster)

    # Calculate the total number of images in user clusters
    m = np.sum(best_user_scene_sum)

    # Calculate the clusterness score (precision)
    # If there are no images in the user clusters, return 0
    if m == 0:
        cluster_score = 0
    else:
        # Precision = true positives / (true positives + false positives)
        # = images in both GT and user cluster / total images in user cluster
        cluster_score = n / m

    return cluster_score


def get_mAA_score(
    best_gt_scene_sum: np.ndarray,
    best_gt_scene: list,
    thresholds: dict,
    dataset: str,
    best_model: list,
    best_err: list,
    skip_top_thresholds: int,
    to_dec: int,
    lt: int,
) -> float:
    """Calculate the mean Average Accuracy (mAA) score across all scenes.

    This function computes the mean Average Accuracy score across all scenes in a dataset.
    It's essentially a recall-like metric that measures how well camera centers are registered
    to their ground truth positions.

    Args:
        best_gt_scene_sum: Array containing the number of images in ground truth scenes.
        best_gt_scene: List of ground truth scene names.
        thresholds: Dictionary of thresholds for each dataset and scene.
        dataset: The dataset name.
        best_model: List of best models for each scene.
        best_err: List of errors for each scene.
        skip_top_thresholds: Number of top thresholds to skip.
        to_dec: Deduction value for the calculation.
        lt: Number of thresholds minus skip_top_thresholds.

    Returns:
        The mean Average Accuracy score, a value between 0 and 1.
    """
    # Total number of images in all ground truth scenes
    n = np.sum(best_gt_scene_sum)

    # Initialize the numerator for the mAA calculation
    a = 0

    # Process each scene
    for i, scene in enumerate(best_gt_scene):
        # Get thresholds for this scene
        ths = thresholds[dataset][scene]

        # Skip if no model was found for this scene
        if len(best_model[i]) < 1:
            continue

        # Create a boolean mask where True indicates errors below the threshold
        # We skip the first skip_top_thresholds values
        tmp = best_err[i][:, skip_top_thresholds:] < np.expand_dims(
            np.asarray(ths[skip_top_thresholds:]), axis=0
        )

        # Count how many cameras are below each threshold, subtract the deduction value,
        # ensure the result is not negative, and add to the running total
        a = a + np.sum(np.maximum(np.sum(tmp, axis=0) - to_dec, 0))

    # Calculate the denominator for the mAA score
    # This represents the maximum possible score
    b = max(0, lt * (n - len(best_gt_scene) * to_dec))

    # Calculate the final mAA score
    # If denominator is 0, return 0 to avoid division by zero
    if b == 0:
        mAA_score = 0
    else:
        mAA_score = a / b

    return mAA_score


def read_mask_csv(mask_filename: str = "split_mask.csv") -> dict:
    """Read the split mask CSV file for IMC2025.

    The mask file contains labels indicating whether each image is in the public or private split.
    This is used to evaluate performance separately on public and private data splits.

    Args:
        mask_filename: Path to the split mask CSV file.

    Returns:
        A nested dictionary with structure {dataset: {scene: {image: mask_value}}},
        where mask_value is a boolean indicating whether the image is in the public split.
    """

    data = {}  # Dictionary to store the mask values
    label_idx = {}  # Dictionary to map column names to indices

    with open(mask_filename, newline="\n") as csvfile:
        csv_lines = csv.reader(csvfile, delimiter=",")

        header = True
        for row in csv_lines:
            if header:
                # Process header row
                header = False
                for i, name in enumerate(row):
                    label_idx[name] = i
                continue

            # Extract data from the current row
            dataset = row[label_idx["dataset"]]
            scene = row[label_idx["scene"]]
            image = row[label_idx["image"]]

            # Convert mask value to boolean (True for public split, False for private split)
            label = row[label_idx["mask"]] == "True"

            # Build the nested dictionary structure
            if dataset not in data:
                data[dataset] = {}

            if scene not in data[dataset]:
                data[dataset][scene] = {}

            data[dataset][scene][image] = label

    return data


def score(
    *,
    gt_csv: str,
    user_csv: str,
    thresholds_csv: str,
    mask_csv: str = None,
    combo_mode: str = "harmonic",
    inl_cf: float = 0,
    strict_cf: float = -1,
    skip_top_thresholds: int = 2,
    to_dec: int = 3,
    verbose: bool = False,
) -> tuple:
    """Compute the IMC 2025 evaluation score.

    This is the main evaluation function for the IMC 2025 competition. It computes scores
    for both the full dataset and the public/private splits if a mask is provided.

    The score combines the mean Average Accuracy (mAA) of registered camera centers and
    the clustering score, which measures how well images are grouped into scenes.

    Args:
        gt_csv: Path to the ground truth CSV file.
        user_csv: Path to the user submission CSV file.
        thresholds_csv: Path to the thresholds CSV file.
        mask_csv: Path to the public/private split mask CSV file. If None, all data is treated as public.
        combo_mode: How to combine mAA and clusterness scores. Options are:
                   "harmonic" (F1 score), "geometric", "arithmetic", "mAA", or "clusterness".
        inl_cf: Inlier confidence factor for camera registration.
        strict_cf: Strict confidence factor for camera registration.
        skip_top_thresholds: Number of top thresholds to skip in mAA calculation.
        to_dec: Deduction value for mAA calculation.
        verbose: Whether to print detailed score information.

    Returns:
        A tuple containing:
        - A tuple of (final_score, final_score_mask_a, final_score_mask_b)
        - A tuple of (scene_score_dict, scene_score_dict_mask_a, scene_score_dict_mask_b)

    Note:
        mask_a refers to the public split, mask_b refers to the private split.
    """

    # Load ground truth and user submission data
    gt_data = read_csv(gt_csv)
    user_data = read_csv(user_csv)

    # Verify that the data is valid
    assert check_data(gt_data, user_data, print_error=True)

    # Load or generate mask for public/private split
    # If mask_csv is None, all data is treated as public
    mask = read_mask_csv(mask_csv) if mask_csv else generate_mask_all_public(gt_data)

    # Calculate the percentage of public images
    # This is used to scale the deduction value for public/private splits
    one_mask = 0  # Count of public images
    all_mask = 0  # Count of all images
    for dataset in mask:
        for scene in mask[dataset]:
            one_mask = one_mask + sum(
                [1 for image in mask[dataset][scene] if mask[dataset][scene][image]]
            )
            all_mask = all_mask + len(mask[dataset][scene])
    pct = one_mask / all_mask  # Percentage of public images

    # Load thresholds for evaluation
    thresholds, th_n = tth_from_csv(thresholds_csv)
    lt = th_n - skip_top_thresholds  # Number of thresholds to use (excluding skipped ones)

    # Initialize arrays to store statistics for each dataset
    # Full dataset statistics
    stat_score = []  # Combined scores
    stat_mAA = []  # mAA scores
    stat_clusterness = []  # Clustering scores

    # Public split statistics (mask_a)
    stat_score_mask_a = []
    stat_mAA_mask_a = []
    stat_clusterness_mask_a = []

    # Private split statistics (mask_b)
    stat_score_mask_b = []
    stat_mAA_mask_b = []
    stat_clusterness_mask_b = []

    # Process each dataset separately
    for dataset in gt_data.keys():
        gt_dataset = gt_data[dataset]  # Ground truth data for this dataset
        user_dataset = user_data[dataset]  # User submission data for this dataset

        lg = len(gt_dataset)  # Number of ground truth scenes
        lu = len(user_dataset)  # Number of user scenes

        # Initialize tables to store results for all possible gt/user scene combinations
        # These tables will have dimensions (num_gt_scenes x num_user_scenes)

        # Tables for full dataset evaluation
        model_table = []  # Store registration models
        err_table = []  # Store registration errors
        mAA_table = np.zeros((lg, lu))  # Store mAA scores
        cluster_table = np.zeros((lg, lu))  # Store cluster overlap counts
        gt_scene_sum_table = np.zeros((lg, lu))  # Store ground truth scene sizes
        user_scene_sum_table = np.zeros((lg, lu))  # Store user scene sizes

        # Tables for public split evaluation (mask_a)
        err_table_mask_a = []  # Store registration errors for public split
        mAA_table_mask_a = np.zeros((lg, lu))  # Store mAA scores for public split
        cluster_table_mask_a = np.zeros((lg, lu))  # Store cluster overlap counts for public split
        gt_scene_sum_table_mask_a = np.zeros(
            (lg, lu)
        )  # Store ground truth scene sizes for public split
        user_scene_sum_table_mask_a = np.zeros((lg, lu))  # Store user scene sizes for public split

        # Tables for private split evaluation (mask_b)
        err_table_mask_b = []
        mAA_table_mask_b = np.zeros((lg, lu))
        cluster_table_mask_b = np.zeros((lg, lu))
        gt_scene_sum_table_mask_b = np.zeros((lg, lu))
        user_scene_sum_table_mask_b = np.zeros((lg, lu))

        # Arrays to store the best results after greedy assignment
        # Best results for full dataset
        best_gt_scene = []  # Ground truth scene names
        best_user_scene = []  # Corresponding user scene names
        best_model = []  # Best registration models
        best_err = []  # Best registration errors
        best_mAA = np.zeros(lg)  # Best mAA scores
        best_cluster = np.zeros(lg)  # Best cluster overlap counts
        best_gt_scene_sum = np.zeros(lg)  # Best ground truth scene sizes
        best_user_scene_sum = np.zeros(lg)  # Best user scene sizes

        # Best results for public split (mask_a)
        best_err_mask_a = []  # Best registration errors for public split
        best_mAA_mask_a = np.zeros(lg)  # Best mAA scores for public split
        best_cluster_mask_a = np.zeros(lg)  # Best cluster overlap counts for public split
        best_gt_scene_sum_mask_a = np.zeros(lg)  # Best ground truth scene sizes for public split
        best_user_scene_sum_mask_a = np.zeros(lg)  # Best user scene sizes for public split

        # Best results for private split (mask_b)
        best_err_mask_b = []
        best_mAA_mask_b = np.zeros(lg)
        best_cluster_mask_b = np.zeros(lg)
        best_gt_scene_sum_mask_b = np.zeros(lg)
        best_user_scene_sum_mask_b = np.zeros(lg)

        # Evaluate all possible ground truth scene to user scene associations
        gt_scene_list = []  # List to store ground truth scene names
        for i, gt_scene in enumerate(gt_dataset.keys()):
            gt_scene_list.append(gt_scene)

            # Initialize arrays to store results for this ground truth scene
            model_row = []  # Registration models for this GT scene with all user scenes
            err_row = []  # Registration errors for this GT scene with all user scenes
            err_row_mask_a = []  # Registration errors for public split
            err_row_mask_b = []  # Registration errors for private split

            user_scene_list = []  # List to store user scene names
            for j, user_scene in enumerate(user_dataset.keys()):
                user_scene_list.append(user_scene)

                # Skip outlier scenes - they don't participate in registration
                if (gt_scene == "outliers") or (user_scene == "outliers"):
                    model_row.append([])
                    err_row.append([])
                    err_row_mask_a.append([])
                    err_row_mask_b.append([])
                    continue

                # Get thresholds for this ground truth scene
                ths = thresholds[dataset][gt_scene]

                # Get camera data for both ground truth and user scenes
                gt_cams = gt_data[dataset][gt_scene]  # Ground truth cameras
                user_cams = user_data[dataset][user_scene]  # User submission cameras

                # Count total number of cameras in ground truth scene
                # These are used as denominators in mAA calculations
                m = len(gt_cams)  # Total number of ground truth cameras
                m_mask_a = np.sum(
                    [mask[dataset][gt_scene][image] for image in mask[dataset][gt_scene].keys()]
                )  # Number of public ground truth cameras
                m_mask_b = np.sum(
                    [not mask[dataset][gt_scene][image] for image in mask[dataset][gt_scene].keys()]
                )  # Number of private ground truth cameras

                # Find cameras that exist in both ground truth and user submission
                # These are the cameras we can use for registration
                good_cams = []
                for image_path in gt_cams.keys():
                    if image_path in user_cams.keys():
                        good_cams.append(image_path)

                # Create masks for public/private split evaluation
                # For each camera, determine if it's in the public split
                good_cams_mask = []
                for image in good_cams:
                    good_cams_mask.append(mask[dataset][gt_scene][image])
                good_cams_mask_a = np.asarray(good_cams_mask)  # Public split mask

                # For each camera, determine if it's in the private split
                good_cams_mask = []
                for image in good_cams:
                    good_cams_mask.append(not mask[dataset][gt_scene][image])
                good_cams_mask_b = np.asarray(good_cams_mask)  # Private split mask

                # Count cameras in each split
                n = len(good_cams)  # Total number of matching cameras
                n_mask_a = np.sum(good_cams_mask_a)  # Number of matching public cameras
                n_mask_b = np.sum(good_cams_mask_b)  # Number of matching private cameras

                # Extract camera centers into matrices for registration
                # These will be used by the Horn's method
                u_cameras = np.zeros((3, n))  # User camera centers
                g_cameras = np.zeros((3, n))  # Ground truth camera centers

                # Fill the camera center matrices
                ii = 0
                for k in good_cams:
                    u_cameras[:, ii] = user_cams[k]["c"]  # User camera center
                    g_cameras[:, ii] = gt_cams[k]["c"]  # Ground truth camera center
                    ii += 1

                # Register camera centers using Horn's method
                # This finds the best similarity transform for each threshold
                model = register_by_Horn(u_cameras, g_cameras, np.asarray(ths), inl_cf, strict_cf)

                # Calculate mAA score for full dataset
                # This measures how well camera centers are registered to ground truth positions
                mAA = mAA_on_cameras(model["err"], ths, m, skip_top_thresholds, to_dec)

                # Calculate mAA score for public split (mask_a)
                # Handle the case where there are no valid cameras or no public cameras
                if (len(model["valid_cams"]) == 0) or (len(good_cams_mask_a) == 0):
                    mAA_mask_a = np.float64(0.0)  # No score if no valid cameras
                else:
                    # Calculate mAA using only public cameras
                    # Note: to_dec is scaled by pct (percentage of public images)
                    mAA_mask_a = mAA_on_cameras(
                        model["err"][
                            good_cams_mask_a[model["valid_cams"]]
                        ],  # Errors for public cameras
                        ths,  # Thresholds
                        m_mask_a,  # Number of public ground truth cameras
                        skip_top_thresholds,  # Number of top thresholds to skip
                        to_dec * pct,  # Scaled deduction value for public split
                    )

                # Calculate mAA score for private split (mask_b)
                # Handle the case where there are no valid cameras or no private cameras
                if (len(model["valid_cams"]) == 0) or (len(good_cams_mask_b) == 0):
                    mAA_mask_b = np.float64(0.0)  # No score if no valid cameras
                else:
                    # Calculate mAA using only private cameras
                    # Note: to_dec is scaled by (1-pct) (percentage of private images)
                    mAA_mask_b = mAA_on_cameras(
                        model["err"][
                            good_cams_mask_b[model["valid_cams"]]
                        ],  # Errors for private cameras
                        ths,  # Thresholds
                        m_mask_b,  # Number of private ground truth cameras
                        skip_top_thresholds,  # Number of top thresholds to skip
                        to_dec * (1 - pct),  # Scaled deduction value for private split
                    )

                # Count total number of images in the user scene
                len_user_scene = len(user_data[dataset][user_scene])

                # Create a mapping of image names to their public/private status
                # This is used to determine which images in the user scene are public/private
                aux_masked = {}
                masked_dataset = mask[dataset]
                for scene in masked_dataset.keys():
                    for image in masked_dataset[scene]:
                        aux_masked[image] = masked_dataset[scene][image]

                # Determine which images in the user scene are in the mask
                # (i.e., which ones have public/private status information)
                user_data_masked = []
                for image in user_data[dataset][user_scene]:
                    if image in aux_masked:
                        user_data_masked.append(aux_masked[image])

                # Count public and private images in the user scene
                len_user_scene_mask_a = np.sum(
                    np.asarray(user_data_masked)
                )  # Number of public images
                len_user_scene_mask_b = np.sum(
                    ~np.asarray(user_data_masked)
                )  # Number of private images

                # Store results for full dataset evaluation
                err_row.append(model["err"])  # Registration errors
                mAA_table[i, j] = mAA  # mAA score
                cluster_table[i, j] = n  # Number of matching cameras (cluster overlap)
                gt_scene_sum_table[i, j] = m  # Number of ground truth cameras
                user_scene_sum_table[i, j] = len_user_scene  # Number of user scene cameras

                # Store registration errors for public split
                # Handle the case where there are no valid cameras or no public cameras
                if (len(model["valid_cams"]) == 0) or (len(good_cams_mask_a) == 0):
                    err_row_mask_a.append(np.zeros((0, th_n)))  # Empty array if no valid cameras
                else:
                    # Store errors for public cameras
                    err_row_mask_a.append(model["err"][good_cams_mask_a[model["valid_cams"]]])

                # Store registration errors for private split
                # Handle the case where there are no valid cameras or no private cameras
                if (len(model["valid_cams"]) == 0) or (len(good_cams_mask_b) == 0):
                    err_row_mask_b.append(np.zeros((0, th_n)))  # Empty array if no valid cameras
                else:
                    # Store errors for private cameras
                    err_row_mask_b.append(model["err"][good_cams_mask_b[model["valid_cams"]]])

                # Store results for public split evaluation
                mAA_table_mask_a[i, j] = mAA_mask_a  # mAA score for public split
                cluster_table_mask_a[i, j] = n_mask_a  # Number of matching public cameras
                gt_scene_sum_table_mask_a[i, j] = m_mask_a  # Number of public ground truth cameras
                user_scene_sum_table_mask_a[i, j] = (
                    len_user_scene_mask_a  # Number of public user scene cameras
                )

                # Store results for private split evaluation
                mAA_table_mask_b[i, j] = mAA_mask_b
                cluster_table_mask_b[i, j] = n_mask_b
                gt_scene_sum_table_mask_b[i, j] = m_mask_b
                user_scene_sum_table_mask_b[i, j] = len_user_scene_mask_b

                # Store the registration model for this ground truth scene and user scene pair
                model_row.append(model)

            model_table.append(model_row)
            err_table.append(err_row)
            err_table_mask_a.append(err_row_mask_a)
            err_table_mask_b.append(err_row_mask_b)

        # Perform greedy assignment of user scenes to ground truth scenes
        # For each ground truth scene, find the best matching user scene
        # based on mAA score (primary) and cluster overlap (secondary)
        for i, gt_scene in enumerate(gt_dataset.keys()):
            # Find the best user scene for this ground truth scene
            # lexsort sorts first by -cluster_table (secondary key) then by -mAA_table (primary key)
            # This prioritizes mAA score, and uses cluster overlap as a tiebreaker
            best_ind = np.lexsort((-mAA_table[i], -cluster_table[i]))[0]

            # Store the best scene pairing
            best_gt_scene.append(gt_scene)  # Ground truth scene
            best_user_scene.append(user_scene_list[best_ind])  # Best matching user scene
            best_model.append(model_table[i][best_ind])  # Registration model for this pairing

            # Store the best results for full dataset evaluation
            best_err.append(err_table[i][best_ind])  # Registration errors
            best_mAA[i] = mAA_table[i, best_ind]  # mAA score
            best_cluster[i] = cluster_table[i, best_ind]  # Cluster overlap count
            best_gt_scene_sum[i] = gt_scene_sum_table[i, best_ind]  # Ground truth scene size
            best_user_scene_sum[i] = user_scene_sum_table[i, best_ind]  # User scene size

            # Store the best results for public split evaluation
            best_err_mask_a.append(
                err_table_mask_a[i][best_ind]
            )  # Registration errors for public split
            best_mAA_mask_a[i] = mAA_table_mask_a[i, best_ind]  # mAA score for public split
            best_cluster_mask_a[i] = cluster_table_mask_a[
                i, best_ind
            ]  # Cluster overlap for public split
            best_gt_scene_sum_mask_a[i] = gt_scene_sum_table_mask_a[
                i, best_ind
            ]  # GT scene size for public split
            best_user_scene_sum_mask_a[i] = user_scene_sum_table_mask_a[
                i, best_ind
            ]  # User scene size for public split

            # Store the best results for private split evaluation
            best_err_mask_b.append(err_table_mask_b[i][best_ind])
            best_mAA_mask_b[i] = mAA_table_mask_b[i, best_ind]
            best_cluster_mask_b[i] = cluster_table_mask_b[i, best_ind]
            best_gt_scene_sum_mask_b[i] = gt_scene_sum_table_mask_b[i, best_ind]
            best_user_scene_sum_mask_b[i] = user_scene_sum_table_mask_b[i, best_ind]

        # Exclude outliers from evaluation
        # Outliers are images that don't belong to any scene and should not be scored
        outlier_idx = -1
        for i, scene in enumerate(best_gt_scene):
            if scene == "outliers":
                outlier_idx = i  # Found the outliers scene
                break

        # If an outliers scene was found, remove it from all arrays
        if outlier_idx > -1:
            # Remove outliers from scene lists
            best_gt_scene.pop(outlier_idx)
            best_user_scene.pop(outlier_idx)
            best_model.pop(outlier_idx)

            # Remove outliers from full dataset arrays
            best_err.pop(outlier_idx)
            best_mAA = np.delete(best_mAA, outlier_idx)
            best_cluster = np.delete(best_cluster, outlier_idx)
            best_gt_scene_sum = np.delete(best_gt_scene_sum, outlier_idx)
            best_user_scene_sum = np.delete(best_user_scene_sum, outlier_idx)

            # Remove outliers from public split arrays
            best_err_mask_a.pop(outlier_idx)
            best_mAA_mask_a = np.delete(best_mAA_mask_a, outlier_idx)
            best_cluster_mask_a = np.delete(best_cluster_mask_a, outlier_idx)
            best_gt_scene_sum_mask_a = np.delete(best_gt_scene_sum_mask_a, outlier_idx)
            best_user_scene_sum_mask_a = np.delete(best_user_scene_sum_mask_a, outlier_idx)

            # Remove outliers from private split arrays
            best_err_mask_b.pop(outlier_idx)
            best_mAA_mask_b = np.delete(best_mAA_mask_b, outlier_idx)
            best_cluster_mask_b = np.delete(best_cluster_mask_b, outlier_idx)
            best_gt_scene_sum_mask_b = np.delete(best_gt_scene_sum_mask_b, outlier_idx)
            best_user_scene_sum_mask_b = np.delete(best_user_scene_sum_mask_b, outlier_idx)

        # Calculate the clusterness score for this dataset
        # This is a precision-like metric: how many images in the user's cluster actually belong to the GT scene
        cluster_score = get_clusterness_score(best_cluster, best_user_scene_sum)  # Full dataset
        cluster_score_mask_a = get_clusterness_score(
            best_cluster_mask_a,
            best_user_scene_sum_mask_a,  # Public split
        )
        cluster_score_mask_b = get_clusterness_score(
            best_cluster_mask_b,
            best_user_scene_sum_mask_b,  # Private split
        )

        # Calculate the mAA score for this dataset
        # This is a recall-like metric: how well camera centers are registered to their ground truth positions
        mAA_score = get_mAA_score(
            best_gt_scene_sum,  # Ground truth scene sizes
            best_gt_scene,  # Ground truth scene names
            thresholds,  # Registration thresholds
            dataset,  # Current dataset
            best_model,  # Best registration models
            best_err,  # Registration errors
            skip_top_thresholds,  # Number of top thresholds to skip
            to_dec,  # Deduction value
            lt,  # Number of thresholds to use
        )  # Full dataset mAA score

        mAA_score_mask_a = get_mAA_score(
            best_gt_scene_sum_mask_a,  # Public ground truth scene sizes
            best_gt_scene,  # Ground truth scene names
            thresholds,  # Registration thresholds
            dataset,  # Current dataset
            best_model,  # Best registration models
            best_err_mask_a,  # Public registration errors
            skip_top_thresholds,  # Number of top thresholds to skip
            to_dec * pct,  # Scaled deduction value for public split
            lt,  # Number of thresholds to use
        )  # Public split mAA score

        mAA_score_mask_b = get_mAA_score(
            best_gt_scene_sum_mask_b,  # Private ground truth scene sizes
            best_gt_scene,  # Ground truth scene names
            thresholds,  # Registration thresholds
            dataset,  # Current dataset
            best_model,  # Best registration models
            best_err_mask_b,  # Private registration errors
            skip_top_thresholds,  # Number of top thresholds to skip
            to_dec * (1 - pct),  # Scaled deduction value for private split
            lt,  # Number of thresholds to use
        )  # Private split mAA score

        # Combine mAA and clusterness scores using the specified method
        # This produces the final score for this dataset
        score = fuse_score(mAA_score, cluster_score, combo_mode)  # Full dataset score
        score_mask_a = fuse_score(
            mAA_score_mask_a, cluster_score_mask_a, combo_mode
        )  # Public split score
        score_mask_b = fuse_score(
            mAA_score_mask_b, cluster_score_mask_b, combo_mode
        )  # Private split score

        # Print detailed score information for this dataset if verbose is True
        # This is useful for debugging and understanding the performance on each dataset
        if verbose:
            # Print full dataset scores
            print(
                f"{dataset}: score={score * 100:.2f}% (mAA={mAA_score * 100:.2f}%, clusterness={cluster_score * 100:.2f}%)"
            )
            # Print public/private split scores if a mask was provided
            if mask_csv:
                print(
                    f"\tPublic split: score={score_mask_a * 100:.2f}% (mAA={mAA_score_mask_a * 100:.2f}%, clusterness={cluster_score_mask_a * 100:.2f}%)"
                )
                print(
                    f"\tPrivate split: score={score_mask_b * 100:.2f}% (mAA={mAA_score_mask_b * 100:.2f}%, clusterness={cluster_score_mask_b * 100:.2f}%)"
                )

        # Store scores for this dataset in the statistics arrays
        # These will be used to calculate the final scores across all datasets

        # Store full dataset scores
        stat_mAA.append(mAA_score)  # mAA score
        stat_clusterness.append(cluster_score)  # Clusterness score
        stat_score.append(score)  # Combined score

        # Store public split scores
        stat_mAA_mask_a.append(mAA_score_mask_a)  # Public split mAA score
        stat_clusterness_mask_a.append(cluster_score_mask_a)  # Public split clusterness score
        stat_score_mask_a.append(score_mask_a)  # Public split combined score

        # Store private split scores
        stat_mAA_mask_b.append(mAA_score_mask_b)
        stat_clusterness_mask_b.append(cluster_score_mask_b)
        stat_score_mask_b.append(score_mask_b)

    # Calculate final scores by averaging across all datasets
    # Full dataset final scores (multiplied by 100 to get percentages)
    final_score = 100 * np.mean(stat_score)  # Final combined score
    final_mAA = 100 * np.mean(stat_mAA)  # Final mAA score
    final_clusterness = 100 * np.mean(stat_clusterness)  # Final clusterness score

    # Public split final scores
    final_score_mask_a = 100 * np.mean(stat_score_mask_a)  # Final public split combined score
    final_mAA_mask_a = 100 * np.mean(stat_mAA_mask_a)  # Final public split mAA score
    final_clusterness_mask_a = 100 * np.mean(
        stat_clusterness_mask_a
    )  # Final public split clusterness score

    # Private split final scores
    final_score_mask_b = 100 * np.mean(stat_score_mask_b)
    final_mAA_mask_b = 100 * np.mean(stat_mAA_mask_b)
    final_clusterness_mask_b = 100 * np.mean(stat_clusterness_mask_b)

    # Print detailed score information if verbose is True
    if verbose:
        print(
            f"Average over all datasets: score={final_score:.2f}% (mAA={final_mAA:.2f}%, clusterness={final_clusterness:.2f}%)"
        )
        if mask_csv:
            print(
                f"\tPublic split: score={final_score_mask_a:.2f}% (mAA={final_mAA_mask_a:.2f}%, clusterness={final_clusterness_mask_a:.2f}%)"
            )
            print(
                f"\tPrivate split: score={final_score_mask_b:.2f}% (mAA={final_mAA_mask_b:.2f}%, clusterness={final_clusterness_mask_b:.2f}%)"
            )

    # Create dictionaries mapping dataset names to their scores
    # These are useful for analyzing performance on individual datasets
    scene_score_dict = {
        dataset: score * 100 for dataset, score in zip(gt_data, stat_score)
    }  # Full dataset scores by dataset

    # Public split scores by dataset (None if no mask was provided)
    scene_score_dict_mask_a = (
        None
        if mask_csv is None
        else {dataset: score * 100 for dataset, score in zip(gt_data, stat_score_mask_a)}
    )

    # Private split scores by dataset (None if no mask was provided)
    scene_score_dict_mask_b = (
        None
        if mask_csv is None
        else {dataset: score * 100 for dataset, score in zip(gt_data, stat_score_mask_b)}
    )

    # Return the final scores and score dictionaries
    # This allows the caller to access both the overall scores and per-dataset scores
    return (
        (final_score, final_score_mask_a, final_score_mask_b),  # Overall scores
        (scene_score_dict, scene_score_dict_mask_a, scene_score_dict_mask_b),  # Per-dataset scores
    )
