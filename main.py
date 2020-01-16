from __future__ import print_function

import cv2
import imutils
import numpy as np
import torch
import os
import torch.nn as nn
from collections import Counter, OrderedDict

from detection.layers.functions.prior_box import PriorBox
from detection.data import cfg
from detection.models.faceboxes import FaceBoxes
from detection.utils.box_utils import decode
from detection.utils.timer import Timer
from detection.utils.nms.py_cpu_nms import py_cpu_nms

from torchvision import models, transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data

device = 'cpu'
if torch.cuda.is_available():
    torch.cuda.current_device()
    device = 'cuda'

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
               'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
               'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
               'del', 'nothing', 'space']
special_classes = {'space': ' ', 'nothing': ''}

# Load weights for FaceBoxes model which is trained on hands
pretrained_dict = torch.load('weights/hand_boxes.pt')
# For some reason in downloaded weights there is prefix "module", so get rid of them
weights = OrderedDict((k.replace('module.', ''), v) for k, v in pretrained_dict.items())

torch.set_grad_enabled(False)

model = FaceBoxes(phase='test', size=None, num_classes=2)
model.load_state_dict(weights)
model.eval()
# print(model)
model = model.to(device)

asl = models.googlenet(pretrained=True)
num_ftrs = asl.fc.in_features
asl.fc = nn.Linear(num_ftrs, len(class_names))
asl.load_state_dict(torch.load('weights/asl_recognition.pt'))
asl.eval()
# print(asl)
asl = asl.to(device)

hand_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def pre_detection_trans(img):
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    return img, scale


def post_detection_trans(out, im_size, treshold = {'acc': .7, 'nms': .6}):
    priorbox = PriorBox(cfg, out[2], im_size, phase='test')
    priors = priorbox.forward()
    priors = priors.to(device)
    loc, conf, _ = out
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.data.cpu().numpy()[:, 1]

    # ignore low scores
    inds = np.where(scores > treshold['acc'])[0]
    boxes = boxes[inds]
    scores = scores[inds]

    # keep top-K before NMS
    order = scores.argsort()[::-1][:5000]
    boxes = boxes[order]
    scores = scores[order]

    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, treshold['nms'])
    return dets[keep, :]


resize = 2
_t = {'detect': Timer(), 'misc': Timer(), 'letter_pred': Timer()}

word_buf = ''
letter_buf = []
cap = cv2.VideoCapture(0)

try:
    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        to_show = frame.copy()
        img = np.float32(to_show)

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        img, scale = pre_detection_trans(img)

        _t['detect'].tic()
        out = model(img)
        _t['detect'].toc()
        _t['misc'].tic()

        dets = post_detection_trans(out, (im_height, im_width), {'acc': .7, 'nms': .6})
        # keep top-K faster NMS
        dets = dets[:750, :]
        _t['misc'].toc()

        for i in range(dets.shape[0]):
    #         get coordinates of rectangle corners
            min_x = max([int(dets[i][0]) - 50, 0])
            min_y = max([int(dets[i][1]) - 50, 0])
            max_x = min([int(dets[i][2]) + 50, frame.shape[1]])
            max_y = min([int(dets[i][3]) + 50, frame.shape[0]])
            cv2.rectangle(to_show, (min_x, max_y), (max_x, min_y), [0, 0, 255], 3)

    #         make a prediction of letter
            _t['letter_pred'].tic()
            hand_img = frame[min_y:max_y, min_x:max_x]
            hand_img = torch.stack([hand_transforms(hand_img)])
            hand_img = hand_img.to(device)
            asl_outputs = asl(hand_img)
            _, preds = torch.max(asl_outputs, 1)
            _t['letter_pred'].toc()

    #         store some last predictions and choose the most frequent one
    #         to decrease level of failures
            letter_buf.append(class_names[preds[0]])
            if len(letter_buf) == 10:
                pred_letter = max(Counter(letter_buf).items(), key=lambda x: x[1])[0]
                if pred_letter in special_classes:
                    word_buf += special_classes[pred_letter]
                elif pred_letter == 'del':
                    word_buf = word_buf[:-1]
                else:
                    word_buf += pred_letter
                letter_buf = []

        #         clear buffer
            if len(word_buf) > 20:
                word_buf = word_buf[-20:]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(to_show, word_buf,
                    (frame.shape[0] - 30 * len(word_buf), 50),
                    font, 2, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('image', to_show)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as ex:
    print(ex)
finally:
    cap.release()
    cv2.destroyAllWindows()

print('Average time for detection: ', _t['detect'].average_time)
print('Average time for post processing of detector output : ', _t['misc'].average_time)
print('Average time for classification: ', _t['letter_pred'].average_time)
