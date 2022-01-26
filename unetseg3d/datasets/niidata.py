import collections
import os

import imageio
import numpy as np
import torch
import nibabel as nib
import json
import math

from unetseg3d.augment import transforms
from unetseg3d.datasets.utils import ConfigDataset, calculate_stats
from unetseg3d.unet3d.utils import get_logger

logger = get_logger('Dataset')

  
class NiiDataset(ConfigDataset):
    def __init__(self, root_dir, phase, transformer_config, expand_dims=True,
                  patch_shape=(80,80,80),atlas_path=None,suffix_raw='_ana_strip_1mm_center_cropped.nii.gz',suffix_label='_seg_ana_1mm_center_cropped.nii.gz',dirpath='',suffix_aux='_seg_tissue_1mm_center_cropped.nii.gz'):
        assert phase in ['train', 'val', 'test']

        self.phase = phase

        # load raw images
        # images_dir = os.path.join(root_dir, 'images')
        # assert os.path.isdir(images_dir)
        self.paths = self._load_files(root_dir,phase, suffix_raw,dirpath)
        self.images = self._load_images(self.paths)
        self.atlas = self._load_nii(atlas_path) if atlas_path else None
        self.file_path = root_dir
        self.expand_dims=expand_dims
        min_value, max_value, mean, std = calculate_stats(self.images)
        logger.info(f'Input stats: min={min_value}, max={max_value}, mean={mean}, std={std}')

        transformer = transforms.get_transformer(transformer_config, min_value=min_value, max_value=max_value,
                                                 mean=mean, std=std)

        # load raw images transformer
        self.raw_transform = transformer.raw_transform()
        
        self.raws = [np.array(self.images[0].shape)]
        self.patch_shape = patch_shape

        self.mask_paths = self._load_files(root_dir,phase, suffix_label,dirpath)
        self.masks = self._load_masks(self.mask_paths)

        self.tissues=None
        if suffix_aux is not None:
            self.tissue_paths = self._load_files(root_dir,phase, suffix_aux,dirpath)
            self.tissues = self._load_masks(self.tissue_paths)
        self.masks_transform = transformer.label_transform()
        self.subjects = self._load_subjects(root_dir,phase)



    def __getitem__(self, idx):
        if idx >= len(self):
            raise StopIteration

        img =self.images[idx]
        if self.phase != 'test':
            mask = self.masks[idx]
            
            if self.tissues is not None:
                tissue = self.tissues[idx]
            if self.expand_dims:
                img = np.expand_dims(img, axis=0)
            if self.atlas is not None:
                if self.tissues is not None:
                    return self.raw_transform(img), self.masks_transform(mask),self.atlas,self.masks_transform(tissue)
                else:
                    return self.raw_transform(img), self.masks_transform(mask),self.atlas
            else:
                return self.raw_transform(img), self.masks_transform(mask)
        else:
            mask = self.masks[idx]
            if self.atlas is not None:
                return self.raw_transform(img), self.masks_transform(mask),self.subjects[idx],self.atlas
            else:
                return self.raw_transform(img), self.masks_transform(mask),self.subjects[idx]

    def __len__(self):
        return len(self.images)

    @classmethod
    def prediction_collate(cls, batch):
        return None

    @classmethod
    def create_datasets(cls, dataset_config, phase):

        phase_config = dataset_config[phase]
        # load data augmentation configuration
        transformer_config = phase_config['transformer']
        # load files to process
        file_paths = phase_config['file_paths']
        dirpath =phase_config.get('dirpath','')
        atlas_path = phase_config.get('atlas_path',None)
        patch_shape = phase_config['slice_builder']['patch_shape']
        suffix_raw = dataset_config.get('suffix_raw','_ana_strip_1mm_center_cropped.nii.gz')
        suffix_label = dataset_config.get('suffix_label','_seg_ana_1mm_center_cropped.nii.gz')
        suffix_aux = dataset_config.get('suffix_aux',None)

        # mirror padding conf
        expand_dims = dataset_config.get('expand_dims', True)
        return [cls(file_paths[0], phase, transformer_config, expand_dims,patch_shape,atlas_path,suffix_raw,suffix_label,dirpath,suffix_aux)]

    @staticmethod
    def _load_nii(path):
        img = nib.load(path).get_fdata()
        return img

    @staticmethod
    def _load_images(paths):
        images = []
        
        for path in paths:
            img = nib.load(path).get_fdata()
            images.append(img)
        return images

    @staticmethod
    def _load_masks(paths):
        images = []
        
        for path in paths:
            img = nib.load(path).get_fdata()
            images.append(img.astype(np.int64))
        return images

    @staticmethod
    def read_pkl(path,phase):
        with open(path,'r') as f:
            dct = json.load(f)
        return dct[phase]

    @staticmethod
    def _load_files(path,phase, suffix,rootdir=''):
        with open(path,'r') as f:
            dct = json.load(f)
        paths = [rootdir+f+suffix for f in dct[phase]]
        return paths

    @staticmethod
    def _load_subjects(path,phase):
        with open(path,'r') as f:
            dct = json.load(f)
        
        paths = [f.split('/')[-1] for f in dct[phase]]
        return paths