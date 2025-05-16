from typing import Dict, List

import numpy as np

from .dataset import Prediction


def array_to_str(array: np.ndarray) -> str:
    """Convert a numpy array to a semicolon-separated string.

    Args:
        array: Numpy array to convert

    Returns:
        Semicolon-separated string representation of the array
    """
    return ";".join([f"{x:.09f}" for x in array])


def none_to_str(n: int) -> str:
    """Create a string of n semicolon-separated 'nan' values.

    Args:
        n: Number of 'nan' values to include

    Returns:
        Semicolon-separated string of 'nan' values
    """
    return ";".join(["nan"] * n)


def create_submission_file(
    samples: Dict[str, List[Prediction]], output_file: str
) -> None:
    """Create a submission file from the predictions.

    Args:
        samples: Dictionary mapping dataset names to lists of Prediction objects
        output_file: Path to the output submission file
        is_train: Whether this is training mode (True) or test mode (False)

    Returns:
        None
    """
    with open(output_file, "w") as f:
        f.write("dataset,scene,image,rotation_matrix,translation_vector\n")
        for dataset in samples:
            for prediction in samples[dataset]:
                cluster_name = (
                    "outliers"
                    if prediction.cluster_index is None
                    else f"cluster{prediction.cluster_index}"
                )
                rotation = (
                    none_to_str(9)
                    if prediction.rotation is None
                    else array_to_str(prediction.rotation.flatten())
                )
                translation = (
                    none_to_str(3)
                    if prediction.translation is None
                    else array_to_str(prediction.translation)
                )
                f.write(
                    f"{prediction.dataset},{cluster_name},{prediction.filename},{rotation},{translation}\n"
                )