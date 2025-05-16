import numpy as np
import cv2
from tqdm import tqdm
from ..features.matching import match_and_verify

def get_default_camera_matrix(img_width, img_height, default_focal_length=1.2):
    """Creates a default camera intrinsic matrix K."""
    f = default_focal_length * max(img_width, img_height)
    cx = img_width / 2.0
    cy = img_height / 2.0
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

def estimate_poses_for_cluster(cluster_image_ids, features, image_dims, matcher, pairwise_matches, min_inlier_matches=15, ransac_threshold=1.5, min_views_for_triangulation=3, min_3d_points_for_pnp=6, pnp_ransac_threshold=5.0, pnp_confidence=0.999):
    """
    Performs simplified incremental SfM for a single cluster.
    Returns a dictionary of {image_id: (R, T, K)} for registered images.
    R is 3x3 rotation, T is 3x1 translation.
    K is the intrinsic matrix used (can be default).
    Returns nan poses for unregistered images in the cluster.
    """
    num_images_in_cluster = len(cluster_image_ids)
    if num_images_in_cluster < 2:
        print(f"Cluster too small ({num_images_in_cluster} images), cannot perform SfM.")
        return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}

    registered_poses = {} # Stores {image_id: (R, T)}
    registered_points_3d = {} # Stores {point_idx: (X, Y, Z, observations)} where observations is {image_id: keypoint_idx}
    point_counter = 0
    processed_images = set()

    # --- 1. Find Best Initial Pair ---
    best_pair = None
    max_inliers = -1

    for i in range(num_images_in_cluster):
        for j in range(i + 1, num_images_in_cluster):
             
             id1, id2 = cluster_image_ids[i], cluster_image_ids[j]
             kps1, descs1 = features.get(id1, (None, None))
             kps2, descs2 = features.get(id2, (None, None))

             if kps1 is None or kps2 is None: continue

             # Reuse matches if available, otherwise compute
             if (id1, id2) in pairwise_matches:
                 matches = pairwise_matches[(id1, id2)]
                 num_inliers = len(matches) # Assuming stored matches are already inliers
             elif (id2, id1) in pairwise_matches:
                  matches = pairwise_matches[(id2, id1)] # Use symmetric entry
                  # Need to swap query/train indices back if stored swapped
                  matches = [(m[1],m[0]) for m in matches]
                  num_inliers = len(matches)
             else:
                 # This shouldn't happen if graph building stored all matches, but as fallback:
                 matches, num_inliers = match_and_verify(kps1, descs1, kps2, descs2, matcher)
                 if matches is None: continue

             if num_inliers > max_inliers and num_inliers >= min_inlier_matches:
                 max_inliers = num_inliers
                 best_pair = (id1, id2, matches) # Store the actual matches

    if best_pair is None:
        print(f"Could not find a suitable initial pair in cluster with {num_images_in_cluster} images.")
        return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}

    id1, id2, initial_matches = best_pair
    print(f"Initializing SfM with pair ({id1}, {id2}) with {max_inliers} matches.")

    kps1, _ = features[id1]
    kps2, _ = features[id2]
    dims1 = image_dims[id1]
    dims2 = image_dims[id2]

    # Get default K matrices
    K1 = get_default_camera_matrix(dims1[0], dims1[1])
    K2 = get_default_camera_matrix(dims2[0], dims2[1])

    # Extract point coordinates for the initial pair matches
    # pts1 = np.float32([kps1[m.queryIdx].pt for m in initial_matches]).reshape(-1, 1, 2)
    # pts2 = np.float32([kps2[m.trainIdx].pt for m in initial_matches]).reshape(-1, 1, 2)

    pts1 = np.float32([kps1[m[0]].pt for m in initial_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kps2[m[1]].pt for m in initial_matches]).reshape(-1, 1, 2)

    # --- 2. Estimate Relative Pose (E -> R, t) ---
    E, mask_e = cv2.findEssentialMat(pts1, pts2, K1, method=cv2.RANSAC, prob=0.999, threshold=ransac_threshold)

    if E is None or mask_e is None:
        print(f"Essential matrix estimation failed for initial pair ({id1}, {id2}).")
        return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}

    num_inliers_e = int(mask_e.sum())
    print(f"Essential matrix inliers: {num_inliers_e}")
    if num_inliers_e < min_inlier_matches: # Need enough support for E
        print(f"Insufficient inliers ({num_inliers_e}) after Essential matrix estimation.")
        return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}


    # Recover relative pose (R, t) from E
    # recoverPose returns the Rotation and Translation vectors for the *second* camera
    # relative to the *first* camera's coordinate system.
    _, R_rel, t_rel, mask_rp = cv2.recoverPose(E, pts1[mask_e.ravel()==1], pts2[mask_e.ravel()==1], K1) # Use K1 (or K2, assumes same intrinsics for simplicity here)

    if R_rel is None or t_rel is None or mask_rp is None:
        print(f"recoverPose failed for initial pair ({id1}, {id2}).")
        return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}

    # --- 3. Set Initial Poses ---
    # First camera is at the origin
    R1 = np.eye(3)
    T1 = np.zeros((3, 1))
    registered_poses[id1] = (R1, T1)
    processed_images.add(id1)

    # Second camera pose is relative to the first
    R2 = R_rel
    T2 = t_rel
    registered_poses[id2] = (R2, T2)
    processed_images.add(id2)

    # --- 4. Triangulate Initial Points ---
    # Projection matrices P = K[R|T]
    P1 = K1 @ np.hstack((R1, T1))
    P2 = K2 @ np.hstack((R2, T2)) # Use K2 here

    # Get the subset of points that were inliers for recoverPose
    inlier_indices_rp = np.where(mask_e.ravel() == 1)[0][mask_rp.ravel() > 0] # Indices into original `initial_matches`
    pts1_rp = np.float32([kps1[initial_matches[i][0]].pt for i in inlier_indices_rp])
    pts2_rp = np.float32([kps2[initial_matches[i][1]].pt for i in inlier_indices_rp])

    if len(pts1_rp) < min_views_for_triangulation:
         print("Not enough points survived recoverPose for triangulation.")
         # Can still proceed maybe, but less robustly. Return failure for now.
         return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}


    # Triangulate points
    points_4d_hom = cv2.triangulatePoints(P1, P2, pts1_rp.T, pts2_rp.T) # Input shapes (2, N)
    points_3d = points_4d_hom[:3] / points_4d_hom[3] # Convert to non-homogeneous
    points_3d = points_3d.T # Shape (N, 3)

    # Store 3D points and their observations
    for i, pt_3d in enumerate(points_3d):
        original_match_idx = inlier_indices_rp[i]
        kp_idx1 = initial_matches[original_match_idx][0]
        kp_idx2 = initial_matches[original_match_idx][1]

        # Basic check for points behind camera (optional, depends on coordinate system)
        # Cheirality check: point must be in front of both cameras
        pt_world = pt_3d.reshape(3, 1)
        pt_cam1 = R1.T @ (pt_world - T1)
        pt_cam2 = R2.T @ (pt_world - T2)
        if pt_cam1[2] > 0 and pt_cam2[2] > 0: # Check if z-coordinate is positive
             registered_points_3d[point_counter] = {
                 'pos': pt_3d,
                 'observations': {id1: kp_idx1, id2: kp_idx2}
             }
             point_counter += 1

    print(f"Triangulated {len(registered_points_3d)} initial 3D points.")
    if not registered_points_3d:
         print("No valid 3D points triangulated, stopping SfM.")
         # Mark initial pair as failed? Or just return no poses? Let's return nan for all.
         return {img_id: (np.full((3, 3), np.nan), np.full((3, 1), np.nan)) for img_id in cluster_image_ids}


    # --- 5. Incremental Registration ---
    remaining_images = [img_id for img_id in cluster_image_ids if img_id not in processed_images]
    # Sort remaining images by number of matches to already registered images (heuristic)
    # This requires efficiently querying matches, pairwise_matches helps here
    images_to_process_queue = sorted(remaining_images, key=lambda img_id: \
                                     sum(len(pairwise_matches.get((img_id, reg_id), []))
                                         for reg_id in processed_images if (img_id, reg_id) in pairwise_matches),
                                     reverse=True)


    print(f"Attempting to register {len(images_to_process_queue)} remaining images...")
    for img_id_new in tqdm(images_to_process_queue, desc="Registering images"):


        kps_new, descs_new = features.get(img_id_new, (None, None))
        if kps_new is None: continue

        dims_new = image_dims[img_id_new]
        K_new = get_default_camera_matrix(dims_new[0], dims_new[1])

        # Find 2D-3D correspondences between the new image and existing 3D points
        points_3d_for_pnp = []
        points_2d_for_pnp = []
        kp_indices_new_for_pnp = [] # Store kp index in new image for potential triangulation later

        # Iterate through registered 3D points
        observed_in_new = [] # Track which 3d points have a match in the new image
        for pt_idx, pt_data in registered_points_3d.items():
            # Check if this 3D point was observed by any *already registered* image
            registered_observers = pt_data['observations'].keys() & processed_images
            if not registered_observers: continue

            # Find matches between the new image and *one* of the registered observers of this 3D point
            # Pick one observer (e.g., the first one)
            observer_id = next(iter(registered_observers))
            kp_idx_observer = pt_data['observations'][observer_id]
            kps_observer, descs_observer = features[observer_id]

            # Check if matches exist between new image and this observer
            current_matches = []
            if (img_id_new, observer_id) in pairwise_matches:
                current_matches = pairwise_matches[(img_id_new, observer_id)]
            elif (observer_id, img_id_new) in pairwise_matches:
                # Need to swap query/train indices
                 swapped_matches = pairwise_matches[(observer_id, img_id_new)]
                 # Format needs care: pairwise_matches stores (kp_idx1, kp_idx2) tuples or Match objects?
                 # Assuming tuple format: (idx_observer, idx_new)
                 current_matches = [(m[1], m[0]) for m in swapped_matches] # Now (idx_new, idx_observer)

            # Find if the specific keypoint kp_idx_observer has a match in current_matches
            found_match = False
            for kp_idx_new, kp_idx_obs in current_matches:
                 if kp_idx_obs == kp_idx_observer:
                     # Found a 2D correspondence for this 3D point
                     points_3d_for_pnp.append(pt_data['pos'])
                     points_2d_for_pnp.append(kps_new[kp_idx_new].pt)
                     kp_indices_new_for_pnp.append(kp_idx_new)
                     observed_in_new.append(pt_idx)
                     found_match = True
                     break # Only need one match per 3D point for PnP list


        num_correspondences = len(points_3d_for_pnp)
        # print(f"Found {num_correspondences} 2D-3D correspondences for image {img_id_new}")

        if num_correspondences < min_3d_points_for_pnp:
            # print(f"Skipping {img_id_new}: Not enough ({num_correspondences}) 2D-3D correspondences for PnP.")
            continue

        # Estimate pose using PnP + RANSAC
        try:
            points_3d_np = np.array(points_3d_for_pnp, dtype=np.float32)
            points_2d_np = np.array(points_2d_for_pnp, dtype=np.float32)

            # distCoeffs can be assumed None or np.zeros((4,1)) if not estimated/known
            dist_coeffs = np.zeros((4,1))

            success, rvec, tvec, inliers_pnp = cv2.solvePnPRansac(
                points_3d_np, points_2d_np, K_new, dist_coeffs, # Use K_new
                iterationsCount=100,
                reprojectionError=pnp_ransac_threshold,
                confidence=pnp_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE # Or other flags like SOLVEPNP_EPNP
            )

            if success and inliers_pnp is not None and len(inliers_pnp) >= min_3d_points_for_pnp:
                R_new, _ = cv2.Rodrigues(rvec) # Convert rotation vector to matrix
                T_new = tvec

                # Add pose to registered list
                registered_poses[img_id_new] = (R_new, T_new)
                processed_images.add(img_id_new)
                print(f"Successfully registered image {img_id_new} ({len(inliers_pnp)} PnP inliers).")

                # --- 6. Optional: Triangulate New Points ---
                # Find matches between this newly registered image and other registered images
                # that observe common features, and triangulate those features if not already 3D points.
                # This makes the reconstruction denser but adds complexity. Skipping for now.
                # Simplified: Update observations for existing points found via PnP
                # for i, pnp_idx in enumerate(inliers_pnp.flatten()):
                #     pt_idx_3d = observed_in_new[pnp_idx] # Find the corresponding 3d point index
                #     kp_idx_new = kp_indices_new_for_pnp[pnp_idx] # Find the corresponding kp index in the new image
                #     if pt_idx_3d in registered_points_3d:
                #          registered_points_3d[pt_idx_3d]['observations'][img_id_new] = kp_idx_new


            else:
                # print(f"PnP failed or insufficient inliers for {img_id_new}.")
                pass # Keep it unregistered

        except cv2.error as e:
            print(f"cv2.solvePnPRansac error for {img_id_new}: {e}")
            continue


    # --- 7. Finalize Poses ---
    final_poses = {}
    for img_id in cluster_image_ids:
        if img_id in registered_poses:
            final_poses[img_id] = registered_poses[img_id] # (R, T)
        else:
            final_poses[img_id] = (np.full((3, 3), np.nan), np.full((3, 1), np.nan))

    print(f"Finished SfM for cluster. Registered {len(registered_poses)} out of {num_images_in_cluster} images.")
    return final_poses

def format_pose(R, T):
    """Formats rotation matrix R and translation vector T into the required submission string format."""
    if R is None or T is None or np.isnan(R).any() or np.isnan(T).any():
        return None, None

    # Ensure R is 3x3 and T is 3x1 or 1x3
    R = np.array(R).reshape(3, 3)
    T = np.array(T).reshape(3,) # Flatten T to 1D array

    return R.flatten(), T

