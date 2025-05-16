#  Copyright [2020] [Michał Tyszkiewicz, Pascal Fua, Eduard Trulls]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import argparse
import os
import warnings

import h5py
import numpy as np
from PIL import ExifTags, Image
from tqdm import tqdm

from .database import COLMAPDatabase, image_ids_to_pair_id


def get_focal(image_path: str, err_on_default: bool = False) -> float:
    """
    Extract focal length from image EXIF data or use a default prior.

    Args:
        image_path: Path to the image file
        err_on_default: If True, raise an error when focal length cannot be found in EXIF

    Returns:
        Estimated focal length in pixels

    Raises:
        RuntimeError: If err_on_default is True and focal length cannot be found in EXIF
    """
    # Open the image to get its dimensions
    image = Image.open(image_path)
    # Get the maximum dimension of the image (width or height)
    max_size = max(image.size)

    # Extract EXIF data from the image
    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # Iterate through all EXIF tags looking for the 35mm focal length
        # Reference: https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        # If we found the 35mm focal length, convert it to pixels based on the image size
        if focal_35mm is not None:
            # Convert the 35mm focal length to the equivalent for this image size
            focal = focal_35mm / 35.0 * max_size

    # If we couldn't find the focal length in EXIF data
    if focal is None:
        if err_on_default:
            # Raise an error if requested
            raise RuntimeError("Failed to find focal length")

        # Use a default focal length prior: 1.2 * max dimension
        # This is a common heuristic when EXIF data is not available
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(db: COLMAPDatabase, image_path: str, camera_model: str) -> int:
    """
    Create a camera entry in the COLMAP database based on image properties.

    Args:
        db: COLMAP database connection
        image_path: Path to the image file
        camera_model: Type of camera model ('simple-pinhole', 'pinhole', 'simple-radial', or 'opencv')

    Returns:
        ID of the created camera
    """
    # Open the image to get its dimensions
    image = Image.open(image_path)
    width, height = image.size

    # Get the focal length from EXIF data or use default estimation
    focal = get_focal(image_path)

    # Set parameters based on the selected camera model
    # Each model has a specific ID number and required parameters
    if camera_model == "simple-pinhole":
        model = 0  # simple pinhole - basic model with single focal length
        # Parameters: [focal_length, principal_point_x, principal_point_y]
        param_arr = np.array([focal, width / 2, height / 2])
    if camera_model == "pinhole":
        model = 1  # pinhole - allows different focal lengths for x and y
        # Parameters: [focal_length_x, focal_length_y, principal_point_x, principal_point_y]
        param_arr = np.array([focal, focal, width / 2, height / 2])
    elif camera_model == "simple-radial":
        model = 2  # simple radial - includes one radial distortion parameter
        # Parameters: [focal_length, principal_point_x, principal_point_y, radial_distortion_1]
        param_arr = np.array([focal, width / 2, height / 2, 0.1])
    elif camera_model == "opencv":
        model = 4  # opencv - full camera model with distortion parameters
        # Parameters: [focal_x, focal_y, cx, cy, k1, k2, p1, p2]
        param_arr = np.array([focal, focal, width / 2, height / 2, 0.0, 0.0, 0.0, 0.0])

    # Add the camera to the database and return the assigned ID
    return db.add_camera(model, width, height, param_arr)


def add_keypoints(
    db: COLMAPDatabase,
    h5_path: str,
    image_path: str,
    img_ext: str,
    camera_model: str,
    single_camera: bool = True,
) -> dict:
    """
    Add keypoints from an HDF5 file to the COLMAP database.

    Args:
        db: COLMAP database connection
        h5_path: Path to the directory containing keypoints.h5
        image_path: Path to the directory containing source images
        img_ext: File extension of the images
        camera_model: Type of camera model to use
        single_camera: If True, use a single camera for all images

    Returns:
        Dictionary mapping filenames to image IDs in the database

    Raises:
        IOError: If an image file specified in the keypoints.h5 doesn't exist
    """
    # Open the HDF5 file containing the keypoints
    keypoint_f = h5py.File(os.path.join(h5_path, "keypoints.h5"), "r")

    # Initialize variables to track camera ID and filename-to-ID mapping
    camera_id = None
    fname_to_id = {}

    # Iterate through all images in the keypoints file with a progress bar
    for filename in tqdm(list(keypoint_f.keys())):
        # Extract keypoints for this image from the HDF5 file
        keypoints = keypoint_f[filename][()]

        # Get the full filename with extension
        fname_with_ext = filename  # + img_ext
        path = os.path.join(image_path, fname_with_ext)

        # Verify that the image file exists
        if not os.path.isfile(path):
            raise IOError(f"Invalid image path {path}")

        # Create a camera if needed
        # If single_camera is True, reuse the same camera for all images (assumes same camera)
        # Otherwise, create a new camera for each image
        if camera_id is None or not single_camera:
            camera_id = create_camera(db, path, camera_model)

        # Add the image to the database and get its ID
        image_id = db.add_image(fname_with_ext, camera_id)
        # Store the mapping from filename to image ID for later use with matches
        fname_to_id[filename] = image_id

        # Add the keypoints for this image to the database
        db.add_keypoints(image_id, keypoints)

    return fname_to_id


def add_matches(db: COLMAPDatabase, h5_path: str, fname_to_id: dict) -> None:
    """
    Add feature matches from an HDF5 file to the COLMAP database.

    Args:
        db: COLMAP database connection
        h5_path: Path to the directory containing matches.h5
        fname_to_id: Dictionary mapping filenames to image IDs in the database

    Warnings:
        Will issue a warning if a pair of images has already been added to the database
    """
    # Open the HDF5 file containing the matches between images
    match_file = h5py.File(os.path.join(h5_path, "matches.h5"), "r")

    # Track which image pairs have already been added to avoid duplicates
    added = set()

    # Calculate the total number of possible image pairs for progress tracking
    # For n images, there are n*(n-1)/2 possible pairs
    n_keys = len(match_file.keys())
    n_total = (n_keys * (n_keys - 1)) // 2

    # Iterate through all image pairs with a progress bar
    with tqdm(total=n_total) as pbar:
        for key_1 in match_file.keys():
            # Get the group containing matches for this image
            group = match_file[key_1]

            # For each image, iterate through all other images it has matches with
            for key_2 in group.keys():
                # Get the database IDs for both images
                id_1 = fname_to_id[key_1]
                id_2 = fname_to_id[key_2]

                # Generate a unique pair ID for this image pair
                # The pair_id is used as a key in the database for this relationship
                pair_id = image_ids_to_pair_id(id_1, id_2)

                # Check if this pair has already been processed
                if pair_id in added:
                    warnings.warn(f"Pair {pair_id} ({id_1}, {id_2}) already added!", stacklevel=1)
                    continue

                # Get the matches for this image pair
                matches = group[key_2][()]

                # Add the matches to the database
                db.add_matches(id_1, id_2, matches)

                # Mark this pair as processed
                added.add(pair_id)

                # Update the progress bar
                pbar.update(1)


def import_into_colmap(
    img_dir: str,
    camera_model: str,
    single_camera: bool = False,
    feature_dir: str = ".featureout",
    database_path: str = "colmap.db",
    img_ext: str = ".jpg",
) -> None:
    """
    Import keypoints and matches into a COLMAP database.

    This is a convenience function that handles the full import process.

    Args:
        img_dir: Path to the directory containing source images
        camera_model: Type of camera model to use
        single_camera: Use a single shared camera for all images if True, or create a new camera for each image otherwise
        feature_dir: Path to the directory with keypoints.h5 and matches.h5
        database_path: Location where the COLMAP .db file will be created
        img_ext: Extension of files in image_path

    Returns:
        None
    """
    # Connect to the database (creates it if it doesn't exist)
    db = COLMAPDatabase.connect(database_path)

    # Create the required tables in the database
    db.create_tables()

    # Add all keypoints to the database and get the mapping from filenames to image IDs
    fname_to_id = add_keypoints(db, feature_dir, img_dir, img_ext, camera_model, single_camera)

    # Add all matches between images to the database
    add_matches(
        db,
        feature_dir,
        fname_to_id,
    )

    # Commit all changes to the database to save them permanently
    db.commit()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", help=("Path to the directory with keypoints.h5 and matches.h5"))
    parser.add_argument("image_path", help="Path to source images")
    parser.add_argument(
        "--image-extension", default=".jpg", type=str, help="Extension of files in image_path"
    )
    parser.add_argument(
        "--database-path",
        default="database.db",
        help="Location where the COLMAP .db file will be created",
    )
    parser.add_argument(
        "--single-camera",
        action="store_true",
        help=(
            "Consider all photos to be made with a single camera (COLMAP "
            "will reduce the number of degrees of freedom"
        ),
    )
    parser.add_argument(
        "--camera-model",
        choices=["simple-pinhole", "pinhole", "simple-radial", "opencv"],
        default="simple-radial",
        help=(
            "Camera model to use in COLMAP. "
            "See https://github.com/colmap/colmap/blob/master/src/base/camera_models.h"
            " for explanations"
        ),
    )

    args = parser.parse_args()

    if args.camera_model == "opencv" and not args.single_camera:
        raise RuntimeError(
            "Cannot use --camera-model=opencv camera without "
            "--single-camera (the COLMAP optimisation will "
            "likely fail to converge)"
        )

    if os.path.exists(args.database_path):
        raise RuntimeError("database path already exists - will not modify it.")

    db = COLMAPDatabase.connect(args.database_path)
    db.create_tables()

    fname_to_id = add_keypoints(
        db,
        args.h5_path,
        args.image_path,
        args.image_extension,
        args.camera_model,
        args.single_camera,
    )
    add_matches(
        db,
        args.h5_path,
        fname_to_id,
    )

    db.commit()
