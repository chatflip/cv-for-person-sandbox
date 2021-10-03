from models.Deeplabv3Mobilenetv3Large import Deeplabv3Mobilenetv3Large
from models.DeepLabv3Resnet101 import DeepLabv3Resnet101
from models.KeypointRCNN import KeypointRCNN


class Detector:
    def __init__(self, algolthm):
        if algolthm == "Deeplabv3Mobilenetv3Large":
            self.model = Deeplabv3Mobilenetv3Large()
        elif algolthm == "KeypointRCNN":
            self.model = KeypointRCNN()
        elif algolthm == "DeepLabv3Resnet101":
            self.model = DeepLabv3Resnet101()

    def preprocess(self, input):
        return self.model.preprocess(input)

    def inference(self, tensor):
        return self.model.inference(tensor)

    def postprocess(self, input, output):
        return self.model.postprocess(input, output)

    def __str__(self):
        return self.model.__class__.__name__
