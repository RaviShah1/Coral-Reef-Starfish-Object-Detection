import numpy as np
import pandas as pd
import cv2
import os

def RecoverGC(sceneRadiance):
    sceneRadiance = sceneRadiance/255.0
    for i in range(3):
        sceneRadiance[:, :, i] =  np.power(sceneRadiance[:, :, i] / float(np.max(sceneRadiance[:, :, i])), 0.7)
    sceneRadiance = np.clip(sceneRadiance*255, 0, 255)
    sceneRadiance = np.uint8(sceneRadiance)
    return sceneRadiance

def apply_GC(df: pd.DataFrame, img_dir: str, out_dir: str, video_id: int, video_frame: int):
    img_path = os.path.join(img_dir, os.path.join(f'video_{video_id}', f'{video_frame}.jpg'))
    img = cv2.imread(img_path)
    sceneRadiance = RecoverGC(img)
    cv2.imwrite(out_dir + 'video_'+str(video_id)+'/' + str(video_frame)+'.jpg', sceneRadiance)
