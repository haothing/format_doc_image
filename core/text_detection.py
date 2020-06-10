import os
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

import native_craft.craft_utils as craft_utils
import native_craft.file_utils as file_utils
import native_craft.imgproc as imgproc
from native_craft.craft import CRAFT
from utils.toolkit import craft_dict_adapt

class TextDetection():
    def __init__(self, net=None, cpu=False, text_threshold=0.5, link_threshold=0.4, low_text=0.4, weight_path='./weights/craft_mlt_25k.pth'):

        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')

        if not net:
            self.net = CRAFT()
            self.net.load_state_dict(craft_dict_adapt(torch.load(weight_path)))
            self.net = self.net.to(self.device)
            self.net.eval()

        self.canvas = 768
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text

    def __net_out(self, text_img):

        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(text_img, 768, interpolation=cv2.INTER_LINEAR, mag_ratio=2)
        ratio_h = ratio_w = 1 / target_ratio

        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)
        x = x.unsqueeze(0).to(self.device)
        y, _ = self.net(x)

        text_map = y[0,:,:,0].cpu().data.numpy()
        link_map = y[0,:,:,1].cpu().data.numpy()
        
        text_bboxes, _ = craft_utils.getDetBoxes(text_map, link_map, self.text_threshold, self.link_threshold, self.low_text, False)
        text_bboxes = craft_utils.adjustResultCoordinates(text_bboxes, ratio_w, ratio_h)

        return text_bboxes, text_map, link_map

    # Use text bound boxes to get more detailed char bound boxes through the network
    # text_bboxes shape: [amount, pos(4), xy(2)]
    def char_bboxes(self, text_img_path, text_bboxes):
        
        if type(text_bboxes) != np.ndarray: text_bboxes = np.array(text_bboxes)
        bboxes = []
        text_plt_img = Image.open(text_img_path)
        for k, box in enumerate(text_bboxes):

            rect = np.append(box.min(axis=0), box.max(axis=0)).tolist()
            text_img = np.array(text_plt_img.crop(rect))
            if len(text_img.shape) == 3 and text_img.shape[0] > 5 and text_img.shape[1] > 5: 
                text_img = cv2.cvtColor(text_img, cv2.COLOR_RGB2GRAY)
            else: continue

            m, r = np.array(text_img.shape).max(), 1
            if np.array(text_img.shape).max() > self.canvas * 0.6:
                r = self.canvas * 0.6 / m
                text_img = cv2.resize(text_img, None, fx=r, fy=r, interpolation=cv2.INTER_LINEAR)

            bg = np.empty([self.canvas, self.canvas], dtype=np.int32)
            bg.fill(255)
            s = (np.array([self.canvas, self.canvas]) - np.array(text_img.shape)) // 2
            e1, e2 = text_img.shape[0] + s[0], text_img.shape[1] + s[1]
            bg[s[0]:e1, s[1]:e2] = text_img

            _, text_map, _ = self.__net_out(cv2.cvtColor(bg.astype(np.uint8), cv2.COLOR_GRAY2RGB))

            img = (255 * text_map.copy()).astype(np.uint8)
            ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3),np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
 
            contours, hierarchy = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            bboxes_intext, rate = [], 0.6
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(contours[i])
                x, y, w, h = x - w * rate / 2 - 2, y - h * rate / 2, w * (1 + rate), h * (1 + rate)
                xy = np.array([x, y])

                fake_box = np.array([xy, xy + [w, 0], xy + [w, h], xy + [0, h]])
                ori_box = (fake_box - (s[1] // 2, s[0] // 2)) * 2 / r + box.min(axis=0)
                bboxes_intext.append(ori_box)

            bboxes += bboxes_intext

        return bboxes

    def bboxes(self, text_img_path):
        
        text_img = imgproc.loadImage(text_img_path)
        text_bboxes, text_map, link_map = self.__net_out(text_img)

        return text_bboxes, text_map, link_map

    def draw_to_file(self, src, dist_dir=None, char_bboxes=False, heatmap=False):

        image_list = []
        if os.path.isfile(src):
            if not dist_dir or os.path.isfile(dist_dir):
                dist_dir = os.path.dirname(src)
            image_list.append(src)
        else:
            if not dist_dir or os.path.isfile(dist_dir):
                dist_dir = os.path.join(os.path.dirname(src), os.path.basename(src) + '_result')
            image_list, _, _ = file_utils.get_files(src)

        os.makedirs(dist_dir, exist_ok=True)
        for k, image_path in enumerate(image_list):

            res_file = os.path.join(dist_dir, os.path.splitext(os.path.basename(image_path))[0]  + '_result.jpg')
            text_bboxes, text_map, link_map = self.bboxes(image_path)

            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            for i in range(len(text_bboxes)):
                box = text_bboxes[i]
                draw.polygon(box.flatten().tolist(), outline="red") 

            if char_bboxes:
                char_bboxes = self.char_bboxes(src, text_bboxes)
                for i in range(len(char_bboxes)):
                    box = char_bboxes[i]
                    draw.polygon(box.flatten().tolist(), outline="green")
            image = image.convert("RGB")

            if heatmap:
                
                image = np.asarray(image)
                text_map = Image.fromarray((text_map * 255).astype(np.uint8), mode='L')
                text_map = text_map.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)
                link_map = Image.fromarray((link_map * 255).astype(np.uint8), mode='L')
                link_map = link_map.resize((image.shape[1], image.shape[0]), resample=Image.NEAREST)

                text_map = np.tile(np.expand_dims(text_map, axis=2), 3)
                link_map = np.tile(np.expand_dims(link_map, axis=2), 3)
                
                image = np.concatenate((image, text_map, link_map), axis=1)
                image = Image.fromarray(np.uint8(image))

            image.save(res_file)

            print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), res_file))