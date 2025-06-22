from typing import Any

import numpy as np
import numpy.typing as npt

from .models.Deeplabv3Mobilenetv3Large import Deeplabv3Mobilenetv3Large
from .models.DeepLabv3Resnet101 import DeepLabv3Resnet101
from .models.FasterrcnnResnet50 import FasterrcnnResnet50
from .models.KeypointRCNN import KeypointRCNN
from .models.MpHolistic import MpHolistic
from .models.MpSelfieSegmentation import MpSelfieSegmentation
from .models.Yunet import Yunet


class Detector:
    """A unified detector class that provides a common interface for various CV models.

    This class acts as a factory and wrapper for different detection and
    segmentation models, providing a consistent API for preprocessing,
    inference, and postprocessing operations.

    Attributes:
        available_models: A list of supported model architectures.
        arch: The architecture name of the selected model.
        model: The instantiated model object.
    """

    available_models = [
        "Deeplabv3Mobilenetv3Large",
        "DeepLabv3Resnet101",
        "FasterrcnnResnet50",
        "KeypointRCNN",
        "MpHolistic",
        "MpSelfieSegmentation",
        "Yunet",
    ]

    def __init__(self, arch: str) -> None:
        """Initialize the Detector with a specified architecture.

        Args:
            arch: The name of the model architecture to use. Must be one of the
                available_models.

        Raises:
            ValueError: If the specified architecture is not supported.
        """
        self.arch = arch
        self.model: Any
        if arch == "Deeplabv3Mobilenetv3Large":
            self.model = Deeplabv3Mobilenetv3Large()
        elif arch == "DeepLabv3Resnet101":
            self.model = DeepLabv3Resnet101()
        elif arch == "FasterrcnnResnet50":
            self.model = FasterrcnnResnet50()
        elif arch == "KeypointRCNN":
            self.model = KeypointRCNN()
        elif arch == "MpHolistic":
            self.model = MpHolistic()
        elif arch == "MpSelfieSegmentation":
            self.model = MpSelfieSegmentation()
        elif arch == "Yunet":
            self.model = Yunet()
        else:
            raise ValueError(f"Invalid architecture: {arch}")

    def preprocess(self, input: npt.NDArray[np.uint8]) -> Any:
        """Preprocess the input image for model inference.

        Args:
            input: Input image as a numpy array with uint8 dtype.

        Returns:
            Preprocessed tensor ready for model inference. The exact type
            depends on the underlying model implementation.
        """
        return self.model.preprocess(input)

    def inference(self, tensor: Any) -> Any:
        """Perform inference on the preprocessed tensor.

        Args:
            tensor: Preprocessed tensor from the preprocess method.

        Returns:
            Raw model output. The exact type and structure depend on the
            underlying model implementation.
        """
        return self.model.inference(tensor)

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: Any
    ) -> npt.NDArray[np.uint8]:
        """Postprocess the model output to generate the final result.

        Args:
            input: Original input image as a numpy array with uint8 dtype.
            output: Raw model output from the inference method.

        Returns:
            Processed output image as a numpy array with uint8 dtype.
            The exact content depends on the model type (e.g., segmentation
            mask, annotated image with bounding boxes, etc.).
        """
        return self.model.postprocess(input, output)  # type: ignore

    @staticmethod
    def get_available_models() -> list[str]:
        """Get the list of available model architectures.

        Returns:
            A list of available model architectures.
        """
        return Detector.available_models

    def __repr__(self) -> str:
        """Return a string representation of the Detector instance.

        Returns:
            A string representation showing the detector's architecture.
        """
        return f"Detector(arch='{self.arch}')"
