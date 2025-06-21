from __future__ import annotations

import os
import sys
import urllib

import cv2
import numpy as np
import numpy.typing as npt

from .BaseModel import BaseModel


class Yunet(BaseModel):
    BASE_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet"
    MODEL_NAME_FLOAT32 = "face_detection_yunet_2023mar.onnx"
    MODEL_NAME_INT8 = "face_detection_yunet_2023mar_int8.onnx"

    def __init__(self, is_int8: bool = False, weight_root: str = "weight") -> None:
        filename = self.MODEL_NAME_INT8 if is_int8 else self.MODEL_NAME_FLOAT32
        model_url = f"{self.BASE_URL}/{filename}"
        weight_path = self.download_weight(weight_root, filename, model_url)
        self.model = cv2.FaceDetectorYN.create(
            model=weight_path,
            config="",
            input_size=(0, 0),
        )

    @staticmethod
    def download_weight(root: str, filename: str, url: str) -> str | None:
        os.makedirs(root, exist_ok=True)
        weight_path = os.path.join(root, filename)
        if os.path.exists(weight_path):
            print(f"File exists: {filename}")
            return weight_path
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, weight_path)
            return weight_path
        except (OSError, urllib.error.HTTPError) as err:
            print(f"ERROR :{err.code}\n{err.reason}")  # type: ignore
            sys.exit()

    def preprocess(self, input: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        height, width = input.shape[:2]
        self.model.setInputSize((width, height))
        return input

    def inference(self, tensor: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        _, faces = self.model.detect(tensor)
        if faces is None:
            faces = np.empty(0, dtype=np.float32)
        return faces  # type: ignore

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        result_image = self.decode_face(input, output)
        return result_image

    @staticmethod
    def decode_face(
        image: npt.NDArray[np.uint8], output: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        for face in output:
            box = list(map(int, face[:4]))
            cv2.rectangle(image, box, (255, 255, 255), 2, cv2.LINE_AA)
            landmarks = list(map(int, face[4 : len(face) - 1]))
            landmarks = np.array(landmarks).reshape(-1, 2)  # type: ignore
            for i in range(landmarks.shape[0]):  # type: ignore
                cv2.circle(image, landmarks[i], 5, (0, 0, 255), -1, cv2.LINE_AA)
        return image
