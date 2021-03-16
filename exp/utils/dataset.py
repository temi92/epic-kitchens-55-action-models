import torch.utils.data
from gulpio2 import GulpDirectory
from torch.utils.data import Dataset
import numpy as np
from abc import ABC
from typing import cast
from collections import OrderedDict
import PIL.Image
class VideoRecord(ABC):
    """
    a video segment with an associated label.
    """

    @property
    def metadata(self):
        raise NotImplementedError()

    @property
    def num_frames(self) -> int:
        raise NotImplementedError()


class VideoDataset(Dataset, ABC):
    @property
    def video_records(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError()

    def load_frames(self, metadata, idx):
        raise NotImplementedError()

class GulpVideoRecord(VideoRecord):
    @property
    def metadata(self):
        return self._metadata_dict

    def __init__(self, gulp_id, gulp_metadata_dict):
        self._metadata_dict = gulp_metadata_dict
        self. gulp_index = gulp_id

    @property
    def num_frames(self):
        return self.metadata["num_frames"]



class CustomDataSet(Dataset):
    def __init__(self, gulped_dir, transform=None):
        self.gd = GulpDirectory(gulped_dir)
        self.items = list(self.gd.merged_meta_dict.items())
        self.transform = transform

    def __getitem__(self, index):
        item_id, item_info = self.items[index]
        images, meta = self.gd[item_id]
        images = list(map(self.convert2PIL, images))
        if self.transform is not None:
            images = self.transform(images)

        return images, meta

    def __len__(self):
        return len(self.items)

    def convert2PIL(self, img):
        return PIL.Image.fromarray(img).convert("RGB")

class TSNDataSet(Dataset):
    """
    designed to interface with epic_training_kitchen_dataset
    """
    def __init__(self, dataset, num_segments=8, seg_length=1, transform=None, random_shift=True, test_mode=True):
        self.dataset = dataset
        self.num_segments = num_segments
        self.seg_length = seg_length
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

    def __getitem__(self, index):
        record = self.dataset.video_records[index]
        
        #TODO implement for training phase as well..
        if self.test_mode:
            segment_start_idxs = self.get_test_indices(record)

        return self.get(record, segment_start_idxs)

    def get(self, record, indices):
        images = self.dataset.load_frames(record, self.get_frame_idxs(indices, record))
  
        
        if self.transform is not None:
            images = self.transform(images)
        return images, record.metadata


    def get_frame_idxs(self, indices, record):
        seg_idxs = []
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.seg_length):
                seg_idxs.append(p)
                if p < record.num_frames:
                    p += 1
        return seg_idxs

    def get_test_indices(self, record):
        tick = (record.num_frames - self.seg_length+1)/float(self.num_segments)
        offsets = np.array([int(tick/2.0+tick*x)for x in range(self.num_segments)])
        return offsets


    def __len__(self):
        return len (self.dataset)


class DatasetWrapper(VideoDataset):
    def __init__(self, gulp_path, transform=None):
        super().__init__()
        self.gulp_dir = GulpDirectory(str(gulp_path.absolute()))
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self._video_records= self._read_video_records(self.gulp_dir.merged_meta_dict)
        #self._read_video_records(self.gulp_dir.merged_meta_dict)
        self._video_records_list = list(self._video_records.values())
     
    @property
    def video_records(self):
        return self._video_records_list

    def load_frames(self, record, indices):
        overall_frames = []
        for i in indices:
            frames = self._sample_video_at_index(cast(GulpVideoRecord, record), i)
            frames = self.transform(frames)
            overall_frames.extend(frames)
        return overall_frames

    def _sample_video_at_index(self, record, index):
        single_frame_slice = slice(index, index+1)
        frame =self.gulp_dir[record.gulp_index, single_frame_slice][0][0]
        return [PIL.Image.fromarray(frame).convert("RGB")]

    def _read_video_records(self, gulp_dir_meta_dict):
        video_records = OrderedDict()
       
        for video_id in gulp_dir_meta_dict:
            meta_dict = gulp_dir_meta_dict[video_id]["meta_data"][0]
            video_records[video_id] = GulpVideoRecord(video_id, meta_dict)
        return video_records

    def __len__(self):
        return len(self._video_records)
