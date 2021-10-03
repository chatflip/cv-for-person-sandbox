from KeypointRCNN import KeypointRCNN


class Detector:
    def __init__(self, algolthm):
        if algolthm == "KeypointRCNN":
            self.model = KeypointRCNN()

    def preprocess(self, input):
        return self.model.preprocess(input)

    def inference(self, tensor):
        return self.model.inference(tensor)

    def postprocess(self, input, output):
        return self.model.postprocess(input, output)

    def __str__(self):
        return self.model.__class__.__name__
