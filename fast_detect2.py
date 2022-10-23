#This file shoud be under yolov5 directory!

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import numpy as np
import asyncio
import time

from PIL import ImageFont, ImageDraw, Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, FixLoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


import allergy


def run():

    model = 'meal.pt'



    alert = " "
    source = str('0')
    white_color = (255,255,255)



    visualize = False
    # Directories
    save_dir = increment_path(Path('runs/detect') / 'exp', exist_ok=False)  # increment run
    (save_dir / 'labels' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = DetectMultiBackend(model, device=device, dnn=False, data=None, fp16=False)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)  # check image size

    # Dataloader
    bs = 1  # bat
    view_img = check_imshow(warn=True)
    dataset = LoadStreams( source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
    bs = len(dataset)

    vid_path, vid_writer = [None] * bs, [None] * bs



    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=False, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            
            p, im0, frame = path[i], im0s[i].copy(), dataset.count
            s += f'{i}: '

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if False else im0  # for save_crop
            annotator = Annotator(im0, line_width=5, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()


                food_list = []
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    food_list.append(names[int(c)])
                
                alert = allergy.detect( food_list )

                # Write results 
                
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = None if False else (names[c] if False else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                

            # Stream results
            im0 = annotator.result()
            
            blue,red,green = (255,0,0),(0,0,255),(0,255,0)  
            font =  cv2.FONT_ITALIC
            

            if alert is None:
                alert = " "
            if alert != " ":
                im0 = cv2.rectangle(im0, (0,50), (1000, 130), white_color ,-1)

            img_pillow = Image.fromarray(im0)
            fontpath = "AppleGothic.ttf"
            font = ImageFont.truetype(fontpath, 25)
            b,g,r,a = 0,255,0,255 
            draw = ImageDraw.Draw(img_pillow, 'RGBA')

            try:
                draw.text((5, 70), alert, font=font, fill=(b,g,r,a))
            except Exception as e:
                print(e)

            im0 = np.array(img_pillow)
            #im0 = cv2.putText(im0, alert, (200,70),font, 1, green, 2, cv2.LINE_AA)

            cv2.imwrite("./static/images/img2.jpg", im0)

            '''
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            '''
run()