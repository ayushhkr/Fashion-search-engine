from __future__ import annotations

import logging
from pathlib import Path
from typing import Final, Union

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn
from torchvision import models, transforms


LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
IMAGE_SIZE: Final[int] = 224
EMBEDDING_DIM: Final[int] = 1280
ImageInput = Union[str, Path, Image.Image]


def build_image_transform() -> transforms.Compose:
    """Create the reusable preprocessing pipeline for EfficientNet-B0 inputs."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class FeatureExtractor:
    """Extract normalized EfficientNet-B0 image embeddings.

    The extractor loads pretrained ImageNet weights once during initialization,
    removes the final classification head, and exposes a simple interface for
    converting either a filesystem image path or a PIL image into a normalized
    1280-dimensional NumPy embedding.
    """

    def __init__(self, device: str | None = None) -> None:
        """Initialize the feature extractor and load the model onto the target device.

        Args:
            device: Optional device override such as ``"cpu"`` or ``"cuda"``.
                When omitted, the extractor uses CUDA if available, otherwise CPU.
        """
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)
        self.transform = build_image_transform()
        self.model = self._load_model().to(self.device)
        self.model.eval()
        LOGGER.info("FeatureExtractor initialized on device=%s", self.device)

    @property
    def embedding_dim(self) -> int:
        """Return the output embedding dimension."""
        return EMBEDDING_DIM

    def _load_model(self) -> nn.Module:
        """Load EfficientNet-B0 with ImageNet weights and remove the classifier."""
        try:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        except Exception as exc:
            LOGGER.exception("Unable to load pretrained EfficientNet-B0 weights.")
            raise RuntimeError(
                "Failed to load pretrained EfficientNet-B0 weights."
            ) from exc

        model.classifier = nn.Identity()
        return model

    def _load_image(self, image: ImageInput) -> Image.Image:
        """Load and validate an image from a path or a PIL image instance."""
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        image_path = Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Expected a file path, received: {image_path}")

        try:
            with Image.open(image_path) as loaded_image:
                return loaded_image.convert("RGB")
        except UnidentifiedImageError as exc:
            LOGGER.warning("Invalid image file: %s", image_path)
            raise ValueError(f"Unsupported or corrupted image file: {image_path}") from exc
        except OSError as exc:
            LOGGER.warning("Unable to read image file: %s", image_path)
            raise ValueError(f"Could not read image file: {image_path}") from exc

    def preprocess(self, image: ImageInput) -> Tensor:
        """Convert an input image into a batched tensor ready for inference."""
        pil_image = self._load_image(image)
        return self.transform(pil_image).unsqueeze(0)

    def extract_embedding(self, image: ImageInput) -> np.ndarray:
        """Extract a normalized 1280-dimensional embedding from an image.

        Args:
            image: A PIL image or a path to an image on disk.

        Returns:
            A NumPy array of shape ``(1280,)`` with ``float32`` dtype.

        Raises:
            FileNotFoundError: If the provided path does not exist.
            ValueError: If the image is invalid, unreadable, or unsupported.
            RuntimeError: If model inference fails.
        """
        input_tensor = self.preprocess(image).to(self.device)

        try:
            with torch.inference_mode():
                embedding = self.model(input_tensor)
                embedding = functional.normalize(embedding, p=2, dim=1)
        except Exception as exc:
            LOGGER.exception("Feature extraction failed during model inference.")
            raise RuntimeError("Failed to extract image embedding.") from exc

        vector = embedding.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if vector.shape != (EMBEDDING_DIM,):
            raise RuntimeError(
                f"Unexpected embedding shape {vector.shape}; expected ({EMBEDDING_DIM},)."
            )
        return vector
