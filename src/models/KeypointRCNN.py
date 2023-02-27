import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision.models.detection import keypointrcnn_resnet50_fpn

from .BaseModel import BaseModel


class KeypointRCNN(BaseModel):
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = keypointrcnn_resnet50_fpn(pretrained=True)
        self.model = model.eval().to(self.device)
        self.POSE_CHAIN = [
            (0, 1),
            (1, 3),
            (0, 2),
            (2, 4),
            (0, 5),
            (5, 7),
            (7, 9),
            (5, 11),
            (11, 13),
            (13, 15),
            (0, 6),
            (6, 8),
            (8, 10),
            (6, 12),
            (12, 14),
            (14, 16),
        ]

    def preprocess(self, input: npt.NDArray[np.uint8]) -> torch.Tensor:
        rgb_image = input[:, :, ::-1].copy()
        tensor = torch.as_tensor(rgb_image, dtype=torch.float) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def inference(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        with torch.inference_mode():  # type: ignore
            output: dict[str, torch.Tensor] = self.model(tensor)[0]
        return output

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: dict[str, torch.Tensor]
    ) -> npt.NDArray[np.uint8]:
        result_image = self.decode_keypoint(input, output)
        return result_image

    def decode_keypoint(
        self, image: npt.NDArray[np.uint8], output: dict[str, torch.Tensor]
    ) -> npt.NDArray[np.uint8]:
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        keypoints = output["keypoints"].cpu()
        num_keypoint = keypoints.shape[1]
        num_target = np.sum(np.where(scores > 0.9, True, False))
        for i in range(num_target):
            if labels[i] == 1:
                keypoint = keypoints[i]
                for j in range(num_keypoint):
                    point = keypoint[j]
                    x, y = int(point[0]), int(point[1])
                    image = cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
                for chain in self.POSE_CHAIN:
                    start = int(keypoint[chain[0], 0]), int(keypoint[chain[0], 1])
                    end = int(keypoint[chain[1], 0]), int(keypoint[chain[1], 1])
                    image = cv2.line(image, start, end, (255, 255, 255), 2)
        return image
