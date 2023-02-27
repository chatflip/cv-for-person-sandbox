from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

from .BaseModel import BaseModel


class MpHolistic(BaseModel):
    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.model = self.mp_holistic.Holistic(static_image_mode=False)
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def preprocess(self, input: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rgb_image: npt.NDArray[np.uint8] = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return rgb_image

    def inference(self, image: npt.NDArray[np.uint8]) -> Any:
        results = self.model.process(image)
        return results

    def postprocess(
        self, input: npt.NDArray[np.uint8], results: Any
    ) -> npt.NDArray[np.uint8]:
        self.mp_drawing.draw_landmarks(
            input,
            results.face_landmarks,
            self.mp_holistic.FACEMESH_TESSELATION,
            self.drawing_spec,
        )
        self.mp_drawing.draw_landmarks(
            input,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.drawing_spec,
        )
        self.mp_drawing.draw_landmarks(
            input,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            self.drawing_spec,
        )
        self.mp_drawing.draw_landmarks(
            input, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS
        )
        return input
