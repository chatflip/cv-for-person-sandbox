import argparse
import sys
import time

import cv2

from src.Detector import Detector


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="demo_webcam")
    parser.add_argument("--arch", type=str, default="MpHolistic")
    parser.add_argument("--camera_width", type=int, default=1280)
    parser.add_argument("--camera_height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--buffersize", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    print(args)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, args.buffersize)
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
        sys.stdout.write("\rFPS: {:.1f}".format(1.0 / interval))
        sys.stdout.flush()
        cv2.imshow("result", result_image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    sys.stdout.write("\r")
    sys.stdout.flush()
    cap.release()


if __name__ == "__main__":
    args = config()
    main(args)
