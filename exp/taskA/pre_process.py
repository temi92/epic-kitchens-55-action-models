import pandas as pd
import sys
from pathlib import Path
sys.path.append("..")
import PIL.Image
import numpy as np

from utils.adapter import DataSetAdapter
import argparse
from gulpio2 import GulpIngestor


parser = argparse.ArgumentParser(description="data preprocessing tool")

parser.add_argument("image_path", type=Path, help="path to directory of images...")
parser.add_argument("labels", type=Path, help="Labels picked data frame")
parser.add_argument("output_folder", type=Path,help="folder to store processed results")

args = parser.parse_args()
pics = args.image_path
labels = args.labels
output_folder = args.output_folder

if not output_folder.exists():
    print ("ouput folder does not exits")
    sys.exit(-1)

#pics = "/home/temi/satis/epic-kitchens-55-action-models/P01/workspace/videos/P01_01"
#labels = pd.read_pickle("/home/temi/satis/epic-kitchens-55-annotations/EPIC_train_action_labels.pkl")

labels = pd.read_pickle(args.labels)
#print(labels)
#grab just a few frames to save time computational time for experimental purposes
labels = labels[:8]

adapter = DataSetAdapter(pics, labels)
ingestor = GulpIngestor(adapter, output_folder, 8, 4)
ingestor()
