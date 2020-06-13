import random
from torchvision import transforms
import torchvision.transforms.functional as TF

class RotationListTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        assert type(x) == list, 'Only list can be passed as argument.'
        angle = random.choice(self.angles)
        return [TF.rotate(item, angle) for item in x]

class ToTensorListTransform:
    def __call__(self, x):
        assert type(x) == list, 'Only list can be passed as argument.'
        
        t = transforms.ToTensor()
        return [t(item) for item in x]