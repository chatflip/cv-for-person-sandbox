import cv2
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class FasterrcnnResnet50:
    def __init__(self, input_half_size=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = model.eval().to(self.device)
        self.input_half_size = input_half_size
        self.person_label = 1

    def preprocess(self, input):
        if self.input_half_size:
            input = cv2.resize(input, None, fx=0.5, fy=0.5)
        rgb_image = input[:, :, ::-1].copy()
        tensor = torch.as_tensor(rgb_image, dtype=torch.float) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    def inference(self, tensor):
        with torch.inference_mode():
            output = self.model(tensor)[0]
        return output

    def postprocess(self, input, output):
        result_image = self.decode_bbox(input, output)
        return result_image

    def decode_bbox(self, image, output):
        boxes = output["boxes"].cpu()
        labels = output["labels"].cpu()
        scores = output["scores"].cpu()
        num_target = np.sum(np.where(scores > 0.9, True, False))
        for i in range(num_target):
            if labels[i] == self.person_label:
                bbox = boxes[i]
                x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 10)
        return image
