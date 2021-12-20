import numpy as np
import pandas as pd
import cv2
import os

def RecoverHE(sceneRadiance):
    for i in range(3):
        sceneRadiance[:, :, i] =  cv2.equalizeHist(sceneRadiance[:, :, i])
    return sceneRadiance

def apply_HE(df: pd.DataFrame, img_dir: str, out_dir: str, video_id: int, video_frame: int):
    img_path = os.path.join(img_dir, os.path.join(f'video_{video_id}', f'{video_frame}.jpg'))
    img = cv2.imread(img_path)
    sceneRadiance = RecoverHE(img)
    cv2.imwrite(out_dir + 'video_'+str(video_id)+'/' + str(video_frame)+'.jpg', sceneRadiance)
