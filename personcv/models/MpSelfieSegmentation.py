from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

from .BaseModel import BaseModel


class MpSelfieSegmentation(BaseModel):
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = self.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def preprocess(self, input: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rgb_image: npt.NDArray[np.uint8] = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return rgb_image

    def inference(self, image: npt.NDArray[np.uint8]) -> Any:
        results = self.model.process(image)
        return results

    def postprocess(
        self, input: npt.NDArray[np.uint8], results: Any
    ) -> npt.NDArray[np.uint8]:
        mask_image = np.where(results.segmentation_mask > 0.1, 255, 0).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        result_image: npt.NDArray[np.uint8] = cv2.addWeighted(
            input, 1.0, mask_image, 1.0, 1.0
        )
        return result_image
