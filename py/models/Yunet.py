import os
import sys
import urllib

import cv2
import numpy as np


class Yunet:
    def __init__(self, weight_root="weight"):
        filename = "yunet.onnx"
        url = "https://raw.github.com/ShiqiYu/libfacedetection.train/master/tasks/task1/onnx/yunet.onnx"
        weight_path = self.download_weight(weight_root, filename, url)
        self.model = cv2.FaceDetectorYN_create(weight_path, "", (0, 0))

    @staticmethod
    def download_weight(root, filename, url):
        os.makedirs(root, exist_ok=True)
        weight_path = os.path.join(root, filename)
        if os.path.exists(weight_path):
            print("File exists: {}".format(filename))
            return weight_path
        try:
            print("Downloading: {}".format(url))
            urllib.request.urlretrieve(url, weight_path)
            return weight_path
        except (OSError, urllib.error.HTTPError) as err:
            print("ERROR :{}".format(err.code))
            print(err.reason)
            sys.exit()

    def preprocess(self, input):
        height, width = input.shape[:2]
        self.model.setInputSize((width, height))
        return input

    def inference(self, tensor):
        _, faces = self.model.detect(tensor)
        if faces is None:
            faces = []
        return faces

    def postprocess(self, input, output):
        result_image = self.decode_face(input, output)
        return result_image

    @staticmethod
    def decode_face(image, output):
        for face in output:
            box = list(map(int, face[:4]))
            cv2.rectangle(image, box, (255, 255, 255), 2, cv2.LINE_AA)
            landmarks = list(map(int, face[4 : len(face) - 1]))
            landmarks = np.array(landmarks).reshape(-1, 2)
            for i in range(landmarks.shape[0]):
                cv2.circle(image, landmarks[i], 5, (0, 0, 255), -1, cv2.LINE_AA)
        return image
