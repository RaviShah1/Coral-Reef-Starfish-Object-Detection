import numpy as np
import pandas as pd
import os
import json

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from typing import List, Dict

def vis_boxes(img_path: str, bboxes: List[Dict[str, int]]):
    coords = []
    for box in bboxes:
        x1 = box['x']
        y1 = box['y']
        x2 = x1 + box['width']
        y2 = y1 + box['height']
        coords.append([x1, y1, x2, y2])
        
    img = Image.open(img_path)
    img1 = img.copy()
    draw = ImageDraw.Draw(img1)
    for elem in coords:
        draw.rectangle(elem, outline='red', width=7)
    
    return img1

def plot_image_with_boxes(df: pd.DataFrame, img_dir: str, video_id: int, video_frame: int):
    img_path = os.path.join(img_dir, os.path.join(f'video_{video_id}', f'{video_frame}.jpg'))
    
    bboxes = list()
    rows = df[(df.video_id == video_id) & (df.video_frame == video_frame)]
    for _, row in rows.iterrows():
        annotations = json.loads(row.annotations.replace("'", '"'))
        for annotation in annotations:
            bboxes.append(annotation)
            
    img = vis_boxes(img_path, bboxes)
    plt.imshow(img)
    plt.show()
    
def plot_image_with_boxes2(df: pd.DataFrame, img_path: str, video_id: int, video_frame: int):    
    bboxes = list()
    rows = df[(df.video_id == video_id) & (df.video_frame == video_frame)]
    for _, row in rows.iterrows():
        annotations = json.loads(row.annotations.replace("'", '"'))
        for annotation in annotations:
            bboxes.append(annotation)
    
    print(img_path)
    img = vis_boxes(img_path, bboxes)
    plt.imshow(img)
    plt.show()
