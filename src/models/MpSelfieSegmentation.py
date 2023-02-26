import cv2
import mediapipe as mp
import numpy as np


class MpSelfieSegmentation:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.selfie_segmentation = mp.solutions.selfie_segmentation
        self.model = self.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def preprocess(self, input):
        rgb_image = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        return rgb_image

    def inference(self, image):
        results = self.model.process(image)
        return results

    def postprocess(self, input, results):
        mask_image = np.where(results.segmentation_mask > 0.1, 255, 0).astype(np.uint8)
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        result_image = cv2.addWeighted(input, 1.0, mask_image, 1.0, 1.0)
        return result_image
