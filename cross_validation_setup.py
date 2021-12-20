"""
Splits a dataframe to a specified cross_validation framework
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

parser = argparse.ArgumentParser(
    description="Cross Validation Setup")
parser.add_argument("-in",
                    "--input_dir",
                    help="The filepath to the input csv.", type=str)
parser.add_argument("-out",
                    "--output_dir",
                    help="The filepath to the output csv.", type=str)
parser.add_argument("-type",
                    "--type",
                    help="The type of cross_validation to use.", type=str)
parser.add_argument("-on",
                    "--on",
                    help="The column to stratisfy on.", type=str)
parser.add_argument("-folds",
                    "--folds",
                    help="The number of folds to create.", type=int)
parser.add_argument("-hold",
                    "--holdout",
                    help="The fold to holdout.", type=int)

args = parser.parse_args()

def cross_validation():
    df = pd.read_csv(args.input_dir)
    df = df[df.annotations!='[]']
    df = df.reset_index(drop=True)
    
    if args.type=='kfold':
        kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
        for f, (t_, v_) in enumerate(kf.split(X=df)):
            df.loc[v_, 'fold'] = f
            
    elif args.type=='skfold' and args.on=='video_id':
        # uses skf to get even distrubution of data from different videos
        kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
        for f, (t_, v_) in enumerate(kf.split(X=df, y=df.video_id)): 
            df.loc[v_, 'fold'] = f
            
    elif args.type=='gkfold' and args.on=='sequence':
        kf = GroupKFold(n_splits=args.folds)
        for f, (t_, v_) in enumerate(kf.split(X=df, y=df.video_id, groups=df.sequence)): 
            df.loc[v_, 'fold'] = f
            
    else:
        raise Exception('Not Implemented')
        
    df.to_csv(args.output_dir, index=False)
    
if __name__ == '__main__':
    cross_validation()
