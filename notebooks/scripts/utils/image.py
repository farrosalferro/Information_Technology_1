import kornia as K
import torch


def load_torch_image(fname: str, device: torch.device = None) -> torch.Tensor:
    """Load an image as a torch tensor.

    Args:
        fname: Path to the image file.
        device: Device to load the image onto. Default is CPU.

    Returns:
        torch.Tensor: Loaded image as a tensor with shape [1, C, H, W].
    """
    if device is None:
        device = torch.device("cpu")

    # Use Kornia's io functionality to load the image with RGB32 format
    # Add an extra dimension at the beginning to get batch dimension [1, C, H, W]
    img = K.io.load_image(fname, K.io.ImageLoadType.RGB32, device=device)[None, ...]
    return img


def get_image_pairs_exhaustive(img_fnames: list[str]) -> list[tuple[int, int]]:
    """Generate all possible image pairs for an exhaustive matching.

    Args:
        img_fnames: List of image file paths.

    Returns:
        list[tuple[int, int]]: List of index pairs for all possible image combinations.
    """
    # Create all possible pairs where i < j to avoid duplicates and self-matches
    index_pairs = []
    for i in range(len(img_fnames)):
        for j in range(i + 1, len(img_fnames)):
            index_pairs.append((i, j))
    return index_pairs
