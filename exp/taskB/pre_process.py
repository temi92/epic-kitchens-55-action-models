import pandas as pd
import sys
sys.path.append("..")
from pathlib import Path
import PIL.Image
import numpy as np
from utils.adapter import CustomAdapter
import argparse
from gulpio2 import GulpIngestor


parser = argparse.ArgumentParser(description="data preprocessing tool for custom dataset ")

parser.add_argument("image_path", type=Path, help="path to directory of images...")
parser.add_argument("output_folder", type=Path,help="folder to store processed results")

args = parser.parse_args()
pics = args.image_path
output_folder = args.output_folder

if not output_folder.exists():
    print ("ouput folder does not exits")
    sys.exit(-1)

adapter = CustomAdapter(pics, output_folder)
ingestor = GulpIngestor(adapter, output_folder, 8, 4)
ingestor()
