import numpy as np
import pandas as pd
import cv2
import os

def clahe_hsv(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h, s, v = hsv_img[:,:,0], hsv_img[:,:,1], hsv_img[:,:,2]
    clahe = cv2.createCLAHE(clipLimit = 15.0, tileGridSize = (20,20))
    v = clahe.apply(v)

    hsv_img = np.dstack((h,s,v))

    rgb = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    
    return rgb

def apply_clahe_hsv(df: pd.DataFrame, img_dir: str, out_dir: str, video_id: int, video_frame: int):
    img_path = os.path.join(img_dir, os.path.join(f'video_{video_id}', f'{video_frame}.jpg'))
    img = cv2.imread(img_path)
    sceneRadiance = clahe_hsv(img)
    cv2.imwrite(out_dir + 'video_'+str(video_id)+'/' + str(video_frame)+'.jpg', sceneRadiance)
