import argparse
import time

import cv2
from src.Detector import Detector


def setup_webcam(args: argparse.Namespace) -> cv2.VideoCapture:
    """Setup webcam with the given arguments.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        cv2.VideoCapture: The webcam object.
    """
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("m", "j", "p", "g"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
    return cap


def inference_webcam(args: argparse.Namespace) -> None:
    """Inference webcam with a given model.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    cap = setup_webcam(args)
    model = Detector(args.arch)

    while cap.isOpened():
        _, frame = cap.read()
        start_time = time.perf_counter()
        tensor = model.preprocess(frame)
        preprocess_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        output = model.inference(tensor)
        inference_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        result_image = model.postprocess(frame, output)
        postprocess_time = time.perf_counter() - start_time

        interval = preprocess_time + inference_time + postprocess_time
        cv2.putText(
            result_image,
            f"FPS: {1.0 / interval:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("result", result_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demo_webcam")
    parser.add_argument("-a", "--arch", type=str, default="MpHolistic")
    parser.add_argument("-c", "--camera_index", type=int, default=0)
    parser.add_argument("-b", "--buffersize", type=int, default=1)
    args = parser.parse_args()
    inference_webcam(args)
