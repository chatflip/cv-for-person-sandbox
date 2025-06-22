import sys
import urllib.request
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


class BaseModel(ABC):
    """Base class for all models."""

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the model."""
        pass

    @abstractmethod
    def preprocess(self, input: npt.NDArray[np.uint8]) -> Any:
        """Preprocess the input image.

        Args:
            input (npt.NDArray[np.uint8]): The input image.

        Returns:
            Any: The preprocessed image.
        """
        pass

    @abstractmethod
    def inference(self, tensor: Any) -> Any:
        """Inference the input image.

        Args:
            tensor (Any): The preprocessed image.

        Returns:
            Any: The inference result.
        """
        pass

    @abstractmethod
    def postprocess(
        self, input: npt.NDArray[np.uint8], output: Any
    ) -> npt.NDArray[np.uint8]:
        """Postprocess and draw the result on the image.

        Args:
            input (npt.NDArray[np.uint8]): The input image.
            output (Any): The inference result.

        Returns:
            npt.NDArray[np.uint8]: The image with the result drawn on it.
        """
        pass

    def download_file(self, url: str, filename: str) -> Path:
        """Download a file from a given URL to a local directory.

        Args:
            url (str): The URL of the file to download.
            filename (str): The name of the file to download.

        Returns:
            Path: The path to the downloaded file.
        """
        root_dir = Path.home() / ".pcv"
        root_dir.mkdir(parents=True, exist_ok=True)

        file_path = root_dir / filename
        if file_path.exists():
            return file_path
        try:
            print(f"Downloading {filename} from {url}")
            urllib.request.urlretrieve(url, file_path)
            return file_path
        except (OSError, urllib.error.URLError) as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)
