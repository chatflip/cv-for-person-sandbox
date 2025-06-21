import argparse
import time
from pathlib import Path

import cv2
from src.Detector import Detector
from tqdm import tqdm


def inference_video(args: argparse.Namespace) -> None:
    """Inference video with a given model.

    Args:
        args (argparse.Namespace): Command line arguments.

    Raises:
        FileNotFoundError: If the input image path does not exist.
    """
    model = Detector(args.arch)
    if not args.input_path.exists():
        raise FileNotFoundError(f"{args.input_path} not exists")
    dst_path = f"{args.input_path.stem}_{args.arch}.mp4"

    cap = cv2.VideoCapture(str(args.input_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc_mp4 = cv2.VideoWriter.fourcc("m", "p", "4", "v")

    writer = cv2.VideoWriter(dst_path, fourcc_mp4, fps, (width, height))

    for _ in tqdm(range(frame_count), f"Running inference: {args.input_path}"):
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.perf_counter()
        tensor = model.preprocess(frame)
        output = model.inference(tensor)
        result_image = model.postprocess(frame, output)
        interval = time.perf_counter() - start_time
        cv2.putText(
            result_image,
            f"FPS: {1.0 / interval:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        writer.write(result_image)
    cap.release()
    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference_image")
    parser.add_argument(
        "-a", "--arch", type=str, default="MpHolistic", help="model architecture"
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default="assets/sample_video.mp4",
        help="input video path",
    )
    args = parser.parse_args()
    inference_video(args)
