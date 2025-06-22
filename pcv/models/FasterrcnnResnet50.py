import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
)

from .BaseModel import BaseModel


class FasterrcnnResnet50(BaseModel):
    def __init__(self, input_half_size: bool = False) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = FasterRCNN_ResNet50_FPN_V2_Weights
        model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model = model.eval().to(self.device)
        self.input_half_size = input_half_size
        self.person_label = 1

    def preprocess(self, input: npt.NDArray[np.uint8]) -> torch.Tensor:
        if self.input_half_size:
            input = cv2.resize(input, None, fx=0.5, fy=0.5)
        rgb_image = input[:, :, ::-1].copy()
        tensor = torch.as_tensor(rgb_image, dtype=torch.float) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def inference(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.inference_mode():
            output: dict[str, torch.Tensor] = self.model(tensor)[0]
        return output

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: dict[str, torch.Tensor]
    ) -> npt.NDArray[np.uint8]:
        result_image = self.decode_bbox(input, output)
        return result_image

    def decode_bbox(
        self, image: npt.NDArray[np.uint8], output: dict[str, torch.Tensor]
    ) -> npt.NDArray[np.uint8]:
        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        num_target = np.sum(np.where(scores > 0.9, True, False))
        for i in range(num_target):
            if labels[i] == self.person_label:
                bbox = boxes[i]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
        return image
