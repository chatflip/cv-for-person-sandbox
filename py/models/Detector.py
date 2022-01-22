from models.Deeplabv3Mobilenetv3Large import Deeplabv3Mobilenetv3Large
from models.DeepLabv3Resnet101 import DeepLabv3Resnet101
from models.FasterrcnnResnet50 import FasterrcnnResnet50
from models.KeypointRCNN import KeypointRCNN
from models.MpHolistic import MpHolistic
from models.MpSelfieSegmentation import MpSelfieSegmentation
from models.Yunet import Yunet


class Detector:
    def __init__(self, arch):
        if arch == "Deeplabv3Mobilenetv3Large":
            self.model = Deeplabv3Mobilenetv3Large()
        elif arch == "DeepLabv3Resnet101":
            self.model = DeepLabv3Resnet101()
        elif arch == "FasterrcnnResnet50":
            self.model = FasterrcnnResnet50()
        elif arch == "KeypointRCNN":
            self.model = KeypointRCNN()
        elif arch == "MpHolistic":
            self.model = MpHolistic()
        elif arch == "MpSelfieSegmentation":
            self.model = MpSelfieSegmentation()
        elif arch == "Yunet":
            self.model = Yunet()

    def preprocess(self, input):
        return self.model.preprocess(input)

    def inference(self, tensor):
        return self.model.inference(tensor)

    def postprocess(self, input, output):
        return self.model.postprocess(input, output)

    def __str__(self):
        return self.model.__class__.__name__
