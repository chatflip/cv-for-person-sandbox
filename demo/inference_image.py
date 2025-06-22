import argparse
import time
from pathlib import Path

import cv2

from pcv.Detector import Detector


def inference_image(args: argparse.Namespace) -> None:
    """Inference image with a given model.

    Args:
        args (argparse.Namespace): Command line arguments.

    Raises:
        FileNotFoundError: If the input image path does not exist.
    """
    model = Detector(args.arch)
    if not args.input_path.exists():
        raise FileNotFoundError(f"{args.input_path} not exists")
    dst_path = f"{args.input_path.stem}_{args.arch}.png"

    image = cv2.imread(str(args.input_path))

    start_time = time.perf_counter()
    tensor = model.preprocess(image)
    preprocess_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    output = model.inference(tensor)
    inference_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    result_image = model.postprocess(image, output)
    postprocess_time = time.perf_counter() - start_time

    interval = preprocess_time + inference_time + postprocess_time
    print(f"preprocess time: {1000 * preprocess_time:.1f}ms")
    print(f"inference time: {1000 * inference_time:.1f}ms")
    print(f"postprocess time: {1000 * postprocess_time:.1f}ms")
    print(f"total time: {1000 * interval:.1f}ms")

    cv2.imwrite(dst_path, result_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference_image")
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        default="MpHolistic",
        choices=Detector.get_available_models(),
        help="model architecture",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default="assets/sample_image.jpg",
        help="input image path",
    )
    args = parser.parse_args()
    inference_image(args)
