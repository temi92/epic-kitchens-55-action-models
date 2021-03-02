import pandas as pd
from pathlib import Path
import  numpy as np
import PIL.Image
from gulpio2.adapters import AbstractDatasetAdapter



def resize_img(imgs):
    for img_path in imgs:
        img = PIL.Image.open(img_path)
        if img is None:
            raise Exception("image not found in {}".format(img_path))
        
        img = np.asarray(img)
        yield img

class DataSetAdapter(AbstractDatasetAdapter):
    """conversion class used for gulping RGB frames"""
    #ref = https://github.com/TwentyBN/GulpIO
    def __init__(self, image_path, annotations):
        self.image_path = image_path
        self.meta_data = self.convert(annotations)

    def convert(self, annotations):
        data = []
        for i, row in annotations.reset_index().iterrows():
            metadata = row.to_dict()
            data.append(metadata)
        return data

    def iter_data(self, slice_element=None):
        """obtain the frame and the corresponding metadata corresponding to the frame"""

        for m in self.meta_data:

            paths = [Path(self.image_path)/f"frame_{idx:010d}.jpg"
                for idx in range(m["start_frame"], m["stop_frame"] + 1)
            ]

            frames = list(resize_img(map(str, paths)))
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
