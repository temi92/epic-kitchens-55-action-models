import pandas as pd
from pathlib import Path
import  numpy as np
import PIL.Image
from gulpio2.adapters import AbstractDatasetAdapter
from gulpio2.utils import resize_images
import os
import glob
import json
import collections

def read_img(imgs):
    for img_path in imgs:
        img = PIL.Image.open(img_path)
        if img is None:
            raise Exception("image not found in {}".format(img_path))
        
        img = np.asarray(img)
        yield img

class DataSetAdapter(AbstractDatasetAdapter):
    #conversion class used for gulping RGB frames with csv labels"""
    #ref = https://github.com/TwentyBN/GulpIO
    def __init__(self, image_path, annotations):
        self.image_path = image_path
        self.meta_data = self.convert(annotations)
        

    def convert(self, annotations):
        data = []
        for i, row in annotations.reset_index().iterrows():
            metadata = row.to_dict()
            data.append(metadata)
        print (data)
        return data

    def iter_data(self, slice_element=None):
        """obtain the frame and the corresponding metadata corresponding to the frame"""

        for m in self.meta_data:

            paths = [Path(self.image_path)/f"frame_{idx:010d}.jpg"
                for idx in range(m["start_frame"], m["stop_frame"] + 1)
            ]

            frames = list(read_img(map(str, paths)))
            m["frame_size"] = frames[0].shape
            m["num_frames"] = len(frames)

            result = {"meta": m, "frames": frames, "id":(self.get_uid(m))}
            yield result
    def get_uid(self, m):
        if "uid" in m :
            return m["uid"]
        else:
            raise ValueError("uid key not present")
            
    def __len__(self):
        return len(self.meta_data)



class CustomAdapter(AbstractDatasetAdapter):
    def __init__(self, image_path, output_folder):

        self.image_path = image_path
        self.output_folder = output_folder
        self.data = CustomAdapter.get_files(self.image_path)
        self.label2idx = self.create_label2idx_dict()
    @staticmethod
    def get_files(folder):
        data = []
        for f in os.listdir(folder):
            sub_folder_data = []
            path = os.path.join(folder, f)
            search_pattern = path + "/*{}".format(".png")
            paths = glob.glob(search_pattern, recursive=True)
            for img_path in paths:
                path = os.path.dirname(img_path)
                label = path.split("/")[-1]
                img_name = os.path.basename(img_path)

                d = collections.OrderedDict()
                d["label"] = label
                d["path"] = os.path.join(path, img_name)
                #data.append({"id":label+ "-" + img_name, "label":label, "path":path})
                sub_folder_data.append(d)
            data.append(sub_folder_data)
        return data

    def iter_data(self, slice_element=None):
        no_frames = 8
        for i, sub_data in enumerate(self.data):
            for j , chunks in enumerate(range(0, len(sub_data), no_frames)):
                    chunk_data  = sub_data[chunks:chunks+no_frames]
                    if len(chunk_data) != no_frames:
                        continue
                    paths  = [d["path"] for d in chunk_data]
                    frames = list(read_img(map(str, paths)))
                    meta = {"label":chunk_data[0]["label"], "idx":self.label2idx[chunk_data[0]["label"]]}
                    result = {"meta":meta, "frames":frames, "id": str(i)+"-"+str(j)}
                    yield result
        self.write_label2idx_dict()

    def create_label2idx_dict(self):
        overall_labels = []
        for sub_data in self.data:
            labels = sorted(set([item['label'] for item in sub_data]))

            overall_labels.extend(labels)
        label2idx = {label: label_counter
                        for label_counter, label in enumerate(overall_labels)}
        return label2idx


    def write_label2idx_dict(self):
        json.dump(self.label2idx,
                  open(os.path.join(self.output_folder, 'label2idx.json'),
                       'w'))

    def __len__(self):
        return len(self.data)
