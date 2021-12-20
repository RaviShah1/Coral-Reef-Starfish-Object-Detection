import numpy as np
import pandas as pd
import cv2
import os
import argparse

from he_enhance import apply_HE
from gc_enhance import apply_GC
from clahe_enhance import apply_CLAHE
from clahe_hsv_enhance import apply_clahe_hsv

parser = argparse.ArgumentParser(
    description="Creates a dataset of enhanced images")
parser.add_argument("-c",
                    "--csv_path",
                    help="Path to the train.csv file.", type=str)
parser.add_argument("-i",
                    "--img_dir",
                    help="Path to the image directory.", type=str)
parser.add_argument("-o",
                    "--out_dir",
                    help="Path to the output directory.", type=str)
parser.add_argument("-t",
                    "--type",
                    help="The type of enhancement to apply.", type=str)
args = parser.parse_args()

if __name__ == '__main__':
    df = pd.read_csv(args.csv_path)
    df = df[df.annotations!='[]']
    df = df.reset_index(drop=True)

    for _, row in df.iterrows():
        if args.type == 'HE':
            apply_HE(df, args.img_dir, args.out_dir, row.video_id, row.video_frame)
        if args.type == 'GC':
            apply_GC(df, args.img_dir, args.out_dir, row.video_id, row.video_frame)
        if args.type == 'clahe':
            apply_CLAHE(df, args.img_dir, args.out_dir, row.video_id, row.video_frame)
        if args.type == 'clahe_hsv':
            apply_clahe_hsv(df, args.img_dir, args.out_dir, row.video_id, row.video_frame)
