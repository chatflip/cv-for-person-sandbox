import cv2
import numpy as np
import torch
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class Deeplabv3Mobilenetv3Large:
    def __init__(self, input_half_size=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model = model.eval().to(self.device)
        self.input_half_size = input_half_size
        self.person_label = 15

    def preprocess(self, input):
        if self.input_half_size:
            input = cv2.resize(input, None, fx=0.5, fy=0.5)
        rgb_image = input[:, :, ::-1].copy()
        tensor = torch.as_tensor(rgb_image, dtype=torch.float) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def inference(self, tensor):
        with torch.inference_mode():
            output = self.model(tensor)["out"].squeeze(0)
        return output

    def postprocess(self, input, output):
        result_image = self.decode_mask(input, output)
        return result_image

    def decode_mask(self, image, output):
        person_mask = output.softmax(dim=0)[self.person_label, :, :].to("cpu")
        without_person_mask = 1.0 - person_mask
        mask_image = np.array(255.0 * without_person_mask, np.uint8)
        if self.input_half_size:
            height, width = image.shape[:2]
            mask_image = cv2.resize(mask_image, (width, height))
        mask_image = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
        result_image = cv2.addWeighted(image, 1.0, mask_image, 1.0, 1.0)
        return result_image
