import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import crnn.lib.models.crnn as crnn
import utils.toolkit as tools

class TextRecognition():

    def __init__(self, net=None, cpu=False, character_set_file='/crnn/lib/config/japanese_char.txt', height=32, 
        weight_path='/weights/text_recognition/checkpoint_500_acc_0.9930.pth'):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.height = height
        self.trans_std = 0.193
        self.trans_mean = 0.588

        char_file = open(character_set_file, "r", encoding="utf-8")
        char_set = ''.join(char_file.read().splitlines())
        self.num_class = len(char_set) + 1
        self.num_hidden = 256
        self.converter = tools.strLabelConverter(char_set)

        if not net:
            self.net = crnn.CRNN(self.height, 1, self.num_class, self.num_hidden)
            self.net.load_state_dict(torch.load(weight_path)['state_dict'])
            self.net = self.net.to(self.device)
            self.net.eval()

    def __init_image(self, pil_image):

        img = pil_image.convert('L')     
        img = img.resize((int(self.height / img.height * img.width), self.height))
        img = (np.array(img).astype(np.float32) / 255. - self.trans_mean) / self.trans_std
        img = np.reshape(img, (1, 1, self.height, -1))

        return torch.from_numpy(img)

    def char(self, text_img):
        pass
        
    def text(self, image_path=None, pil_image=None):

        assert image_path != None or pil_image != None, 'input image will be not none.'

        input_torch, img_list = None, []
        if pil_image == None:

            if os.path.isfile(image_path):
                img_list.append(image_path)
            else:
                img_list, _, _ = tools.get_files(image_path)

            assert len(img_list) > 0, 'no image found.'
            for image_path in img_list:
                pil_img = Image.open(image_path)
                if input_torch == None:
                    input_torch = self.__init_image(pil_img)
                else:
                    input_torch = torch.cat((input_torch, self.__init_image(pil_img)))
        else:
            if type(pil_image) != list:
                img_list.append(pil_image)
            else:
                img_list = pil_image

            assert len(img_list) > 0, 'no image found.'
            for pil_img in img_list:
                if input_torch == None: 
                    input_torch = self.__init_image(pil_img)
                else:
                    input_torch = torch.cat((input_torch, self.__init_image(pil_img)))


        chunks = torch.chunk(input_torch, len(img_list) // 100 + 1)
        result_preds = []
        with torch.no_grad():
            for input_torch in chunks:
                input_torch = input_torch.to(self.device)
                preds = self.net(input_torch).cpu()
                preds_size = torch.IntTensor([preds.size(0)] * input_torch.size(0))

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)

                #print(preds.size())
                #print(preds_size.size())

                result_preds += self.converter.decode(preds.data, preds_size.data, raw=False)

        return result_preds