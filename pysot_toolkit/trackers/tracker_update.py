from __future__ import absolute_import

import numpy as np
import math
import torchvision.transforms.functional as tvisf
import cv2
import torch
import torch.nn.functional as F
import time


class Tracker(object):

    def __init__(self, name, net, instance_size=256, search_factor=4, exemplar_size=128, temp_factor=2,
                 window_penalty=0.49, frequency=100, conf_threshold=0.99):
        self.name = name
        self.net = net
        self.search_factor = search_factor
        self.temp_factor = temp_factor
        self.window_penalty = window_penalty
        self.exemplar_size = exemplar_size
        self.instance_size = instance_size
        self.net = self.net
        self.update_frequency = frequency
        self.conf_threshold = conf_threshold

    def _convert_score(self, score):
        score = score.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 0].cpu().numpy()
        return score

    def _convert_bbox(self, delta):

        delta = delta.permute(2, 1, 0).contiguous().view(4, -1)
        delta = delta.data.cpu().numpy()

        return delta

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: rgb based image
            pos: center position
            model_sz: exemplar size
            original_sz: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                       int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        im_patch = torch.from_numpy(im_patch)
        im_patch = im_patch.cuda()
        return im_patch

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        tic = time.time()
        hanning = np.hanning(32)
        window = np.outer(hanning, hanning)
        self.window = window.flatten()
        # Initialize
        self.initialize_features()
        bbox = info['init_bbox']
        self.center_pos = np.array([bbox[0] + bbox[2] / 2,
                                    bbox[1] + bbox[3] / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + (self.temp_factor - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_z = self.size[1] + (self.temp_factor - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(image, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(image, self.center_pos,
                                    self.exemplar_size,
                                    s_z, self.channel_average)

        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = False
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        # initialize template feature
        self.net.template(z_crop)
        out = {'time': time.time() - tic}
        return out

    @torch.no_grad()
    def update_template(self, image, bbox):
        center_pos = np.array([bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2])
        size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = size[0] + (self.temp_factor - 1) * ((size[0] + size[1]) * 0.5)
        h_z = size[1] + (self.temp_factor - 1) * ((size[0] + size[1]) * 0.5)
        s_z = math.ceil(math.sqrt(w_z * h_z))
        # calculate channle average
        channel_average = np.mean(image, axis=(0, 1))
        # get crop
        z_crop = self.get_subwindow(image, center_pos,
                                    self.exemplar_size,
                                    s_z, channel_average)
        # normalize
        z_crop = z_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        z_crop[0] = tvisf.normalize(z_crop[0], self.mean, self.std, self.inplace)
        # update template feature
        self.net.template_new(z_crop)

    def track(self, image, frame_idx):
        # calculate x crop size
        w_x = self.size[0] + (self.search_factor - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (self.search_factor - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))

        # get crop
        x_crop = self.get_subwindow(image, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)

        # normalize
        x_crop = x_crop.float().mul(1.0 / 255.0).clamp(0.0, 1.0)
        x_crop[0] = tvisf.normalize(x_crop[0], self.mean, self.std, self.inplace)

        # track
        outputs = self.net.track(x_crop)
        score = self._convert_score(outputs['pred_logits'])
        pred_bbox = self._convert_bbox(outputs['pred_boxes'])

        # window penalty
        pscore = score * (1 - self.window_penalty) + \
                 self.window * self.window_penalty

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * s_x
        cx = bbox[0] + self.center_pos[0] - s_x / 2
        cy = bbox[1] + self.center_pos[1] - s_x / 2
        width = bbox[2]
        height = bbox[3]

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, image.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        # if score[best_idx] < conf_score:
        #     print(score[best_idx], frame_idx, 'low confidence, frame_idx')
        if frame_idx % self.update_frequency == 0 and pscore[best_idx] > self.conf_threshold:  # default:200, 0.9
            print(' update frame_idx, conf_score', frame_idx, pscore[best_idx])
            self.update_template(image, bbox)

        out = {'target_bbox': bbox,
               'best_score': pscore}
        return out
