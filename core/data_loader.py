import os
import re
import itertools
import numpy as np

import scipy.io as scio
from core.gaussian import GaussianTransformer
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms.functional as TF

class SynthData(VisionDataset):

    def __init__(self, root, target_size=768, ground_true_file=None, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.target_size = target_size
        self.gaussianTransformer = GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)
        self.root = root

        if ground_true_file == None: ground_true_file = 'gt.mat'
        gt = scio.loadmat(os.path.join(root, ground_true_file))
        self.charbox = gt['charBB'][0]
        self.image = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

    def __getitem__(self, index):

        img_path = os.path.join(self.root, self.image[index][0])
        image = Image.open(img_path).convert('RGB')
        wh = np.array(image).shape

        char_bboxes, words  = self.load_char_bboxes(index)
        region_scores = np.zeros((wh[0], wh[1]), dtype=np.float32)
        affinity_scores = np.zeros((wh[0], wh[1]), dtype=np.float32)
        affinity_bboxes = []
        if len(char_bboxes) > 0:
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, char_bboxes)
            affinity_scores, affinity_bboxes = self.gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                          char_bboxes,
                                                                                          words)

        m = max(wh[:2])
        image = TF.resized_crop(image, 0, 0, m, m, self.target_size)
        region_scores = TF.resized_crop(Image.fromarray(region_scores, mode='L'), 0, 0, m, m, self.target_size // 2)
        affinity_scores = TF.resized_crop(Image.fromarray(affinity_scores, mode='L'), 0, 0, m, m, self.target_size // 2)
        #image = TF.resize(image, self.target_size)
        #region_scores = TF.resize(Image.fromarray(region_scores, mode='L'), self.target_size // 2)
        #affinity_scores = TF.resize(Image.fromarray(affinity_scores, mode='L'), self.target_size // 2)

        if self.transform is not None:
            images = self.transform([image, region_scores, affinity_scores])   

        return images[0], images[1], images[2]

    def __len__(self):
        return len(self.imgtxt)

    def get_imagename(self, index):
        return self.image[index][0]

    def load_char_bboxes(self, index):

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        chars = self.charbox[index].transpose((2, 1, 0))

        char_bboxes = []
        total = 0
        for i in range(len(words)):
            bboxes = chars[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            char_bboxes.append(bboxes)

        return char_bboxes, words