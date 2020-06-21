import random
import numpy as np 

from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

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

class RandomCompressTransform:
    def __init__(self, scale_height=None, scale_width=None):
        
        assert scale_height != None or scale_width != None, 'Need scale_height or scale_width as argument.'
        assert scale_height == None or type(scale_height) == tuple, 'Only tuple can be passed as argument.'
        assert scale_width == None or type(scale_width) == tuple, 'Only tuple can be passed as argument.'

        self.scale_height = scale_height
        self.scale_width = scale_width

    def __call__(self, x):

        w, h = x.width, x.height
        if self.scale_height == None:
            size_h = h
        else:
            size_h = int((np.random.random(1).item() * (self.scale_height[1] - self.scale_height[0]) + self.scale_height[0]) * h)

        if self.scale_width == None:
            size_w = w
        else:
            size_w = int((np.random.random(1).item() * (self.scale_width[1] - self.scale_width[0]) + self.scale_width[0]) * w)

        image = ImageOps.expand(x.resize((size_w, size_h)), border=(0, 0, w - size_w, h - size_h), fill=0)

        return image