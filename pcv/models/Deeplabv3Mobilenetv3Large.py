from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from .BaseModel import BaseModel


class Deeplabv3Mobilenetv3Large(BaseModel):
    def __init__(self, input_half_size: bool = True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        model = deeplabv3_mobilenet_v3_large(weights=weights)
        self.model = model.eval().to(self.device)
        self.input_half_size = input_half_size
        self.person_label = 15

    def preprocess(self, input: npt.NDArray[np.uint8]) -> torch.Tensor:
        if self.input_half_size:
            input = cv2.resize(input, None, fx=0.5, fy=0.5)
        rgb_image = input[:, :, ::-1].copy()
        tensor = torch.as_tensor(rgb_image, dtype=torch.float) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def inference(self, tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():  # type: ignore
            output: torch.Tensor = self.model(tensor)["out"].squeeze(0)
        return output

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: torch.Tensor
    ) -> npt.NDArray[np.uint8]:
        result_image = self.decode_mask(input, output)
        return result_image

    def decode_mask(
        self, image: npt.NDArray[np.uint8], output: torch.Tensor
    ) -> npt.NDArray[np.uint8]:
        person_mask = output.softmax(dim=0)[self.person_label, :, :].to("cpu")
        without_person_mask = 1.0 - person_mask
        mask_image = np.array(255.0 * without_person_mask, np.uint8)
        if self.input_half_size:
            height, width = image.shape[:2]
            mask_image = cv2.resize(mask_image, (width, height))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        result_image: npt.NDArray[np.uint8] = cv2.addWeighted(
            image, 1.0, mask_image, 1.0, 1.0
        )
        return result_image
