import argparse
import os
import time

import cv2
from models.Detector import Detector


def opt():
    parser = argparse.ArgumentParser(description="mbg")
    parser.add_argument("--algorithm", type=str, default="KeypointRCNN")
    parser.add_argument("--input_path", type=str, default="data/sample.jpg")
    args = parser.parse_args()
    return args


def main(args):
    print(args)
    model = Detector(args.algorithm)
    assert os.path.exists(args.input_path), f"{args.input_path} not exists"
    image = cv2.imread(args.input_path)

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
    print(f"preprocess_time: {1000 * preprocess_time:.1f}ms")
    print(f"inference_time: {1000 * inference_time:.1f}ms")
    print(f"postprocess_time: {1000 * postprocess_time:.1f}ms")
    print(f"FPS: {1.0 / interval:.1f}")

    dst_path = os.path.join("data", f"{str(model)}.png")
    cv2.imwrite(dst_path, result_image)


if __name__ == "__main__":
    args = opt()
    main(args)
