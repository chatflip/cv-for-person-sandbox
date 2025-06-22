"""Common fixtures for all test modules."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import pytest


@pytest.fixture(scope="session")
def assets_dir() -> Path:
    """Fixture to provide the assets directory path."""
    return Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def sample_image_path(assets_dir: Path) -> str:
    """Fixture to provide the path to the sample image."""
    image_path = assets_dir / "sample_image.jpg"
    assert image_path.exists(), f"Sample image not found at {image_path}"
    return str(image_path)


@pytest.fixture(scope="session")
def sample_video_path(assets_dir: Path) -> str:
    """Fixture to provide the path to the sample video."""
    video_path = assets_dir / "sample_video.mp4"
    assert video_path.exists(), f"Sample video not found at {video_path}"
    return str(video_path)


@pytest.fixture
def sample_image(sample_image_path: str) -> npt.NDArray[np.uint8]:
    """Fixture to load the sample image."""
    image = cv2.imread(sample_image_path)
    assert image is not None, f"Failed to load image from {sample_image_path}"
    return image  # type: ignore[return-value]


@pytest.fixture
def sample_image_rgb(sample_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Fixture to provide the sample image in RGB format."""
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)  # type: ignore[return-value]


@pytest.fixture
def sample_image_gray(sample_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Fixture to provide the sample image in grayscale format."""
    return cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)  # type: ignore[return-value]
