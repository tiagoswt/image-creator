"""
Image loading utility for handling PIL images and file paths
"""

from PIL import Image
from typing import Union


def load_img(image: Union[Image.Image, str], output_type: str = "pil") -> Image.Image:
    """
    Load an image from various sources and return as PIL Image.

    Args:
        image: Either a PIL Image object or a file path string
        output_type: Output format (default: "pil")

    Returns:
        PIL Image object
    """
    if isinstance(image, str):
        # Load from file path
        img = Image.open(image)
    elif isinstance(image, Image.Image):
        # Already a PIL Image
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    return img
