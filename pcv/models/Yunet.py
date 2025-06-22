import cv2
import numpy as np
import numpy.typing as npt

from .BaseModel import BaseModel


class Yunet(BaseModel):
    """Yunet model for face detection.

    Attributes:
        BASE_URL: The base URL of the model.
        MODEL_NAME_FLOAT32: The name of the float32 model.
        MODEL_NAME_INT8: The name of the int8 model.
    """

    BASE_URL = "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet"
    MODEL_NAME_FLOAT32 = "face_detection_yunet_2023mar.onnx"
    MODEL_NAME_INT8 = "face_detection_yunet_2023mar_int8.onnx"

    def __init__(
        self,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        is_int8: bool = False,
    ) -> None:
        """Initialize the Yunet model.

        Args:
            conf_threshold (float, optional): The confidence threshold. Defaults to 0.6.
            nms_threshold (float, optional): The NMS threshold. Defaults to 0.3.
            top_k (int, optional): The top k detections. Defaults to 5000.
            is_int8 (bool, optional): Whether to use the int8 model. Defaults to False.
        """
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.is_int8 = is_int8
        filename = self.MODEL_NAME_INT8 if is_int8 else self.MODEL_NAME_FLOAT32
        model_url = f"{self.BASE_URL}/{filename}"
        onnx_path = self.download_file(model_url, filename)
        self.model = cv2.FaceDetectorYN.create(
            model=str(onnx_path),
            config="",
            input_size=(0, 0),
        )

    def preprocess(self, input: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """Preprocess the input image.

        Args:
            input (npt.NDArray[np.uint8]): The input image.

        Returns:
            npt.NDArray[np.uint8]: The preprocessed image.
        """
        height, width = input.shape[:2]
        self.model.setInputSize((width, height))
        return input

    def inference(self, tensor: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Inference the input image using the model.

        Args:
            tensor (npt.NDArray[np.uint8]): The input image.

        Returns:
            npt.NDArray[np.float32]: The inference result.
        """
        _, faces = self.model.detect(tensor)
        if faces is None:
            faces = np.empty(0, dtype=np.float32)  # type: ignore[unreachable]
        return faces  # type: ignore

    def postprocess(
        self, input: npt.NDArray[np.uint8], output: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """PostProcess and draw the faces on the image.

        Args:
            input (npt.NDArray[np.uint8]): The input image.
            output (npt.NDArray[np.float32]): The inference result.

        Returns:
            npt.NDArray[np.uint8]: The image with the faces drawn on it.
        """
        result_image = self.decode_face(input, output)
        return result_image

    @staticmethod
    def decode_face(
        image: npt.NDArray[np.uint8], output: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """Decode the faces from the inference result and draw them on the image.

        Args:
            image (npt.NDArray[np.uint8]): The input image.
            output (npt.NDArray[np.float32]): The inference result.

        Returns:
            npt.NDArray[np.uint8]: The image with the faces drawn on it.
        """
        for face in output:
            box = list(map(int, face[:4]))
            cv2.rectangle(image, box, (255, 255, 255), 2, cv2.LINE_AA)
            landmarks = list(map(int, face[4 : len(face) - 1]))
            landmarks_array = np.array(landmarks).reshape(-1, 2)
            for i in range(landmarks_array.shape[0]):
                point = (int(landmarks_array[i, 0]), int(landmarks_array[i, 1]))
                cv2.circle(image, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
        return image

    def __repr__(self) -> str:
        """Return a string representation of the Yunet model.

        Returns:
            str: The string representation of the Yunet model.
        """
        return (
            "Yunet("
            f"conf_threshold={self.conf_threshold}, "
            f"nms_threshold={self.nms_threshold}, "
            f"top_k={self.top_k}, "
            f"is_int8={self.is_int8}"
            ")"
        )
