import dataclasses
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Prediction:
    """Dataclass to store predictions for each image.

    Attributes:
        image_id: A unique identifier for the row -- unused otherwise. Used only on the hidden test set.
        dataset: Name of the dataset this image belongs to
        filename: Image filename
        cluster_index: Index of the image grouping (None if not clustered)
       rotation: 3x3 rotation matrix
        translation: 3D translation vector
    """

    image_id: Optional[str] = None
    dataset: str = ""
    filename: str = ""
    cluster_index: Optional[int] = None
    rotation: Optional[np.ndarray] = None
    translation: Optional[np.ndarray] = None


def load_dataset(
    data_dir: str,
) -> Dict[str, List[Prediction]]:
    """Load the competition CSV file and organize images by dataset name.

    Args:
        data_dir: Path to the competition data directory
        is_train: Whether this is training mode (True) or test mode (False)

    Returns:
        Dictionary mapping dataset names to lists of Prediction objects
    """
    csv_filepath = os.path.join(
        data_dir, "train_labels.csv"
    )

    samples = {}
    competition_data = pd.read_csv(csv_filepath)
    for _, row in competition_data.iterrows():
        # Note: For the test data, the "scene" column has no meaning, and the rotation_matrix and translation_vector columns are random.
        if row.dataset not in samples:
            samples[row.dataset] = []
        samples[row.dataset].append(
            Prediction(
                image_id=None, dataset=row.dataset, filename=row.image
            )
        )
    return samples
