## Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import cv2
import math
import random
import numpy as np

from skimage import feature, exposure
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader, make_dataset


from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.p = 16
        self.num_tokens = int(196 * (1 - 0.85))

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def mask_visualizing(self, sample, target, mask_preds, path="demo_mask"):
        img = cv2.cvtColor(np.asarray(sample),cv2.COLOR_RGB2BGR)
        h = w = img.shape[1] // self.p
        img = img.reshape((h, self.p, w, self.p, 3))
        img = np.einsum('hpwqc->hwpqc', img)

        ids = np.argsort(mask_preds)
        for id in ids[49:]:
            img[int(id/14), int(id%14), :, :, :] = 125
        img = np.einsum('hwpqc->hpwqc', img)
        img = img.reshape((h * self.p, w * self.p, 3))
        cv2.imwrite(os.path.join(path, str(target) + ".png"), img)

    def statistic_masking(self, sample):
        img = cv2.cvtColor(np.asarray(sample),cv2.COLOR_RGB2BGR)
        h = w = img.shape[1] // self.p
        img = img.reshape((h, self.p, w, self.p, 3))
        img = np.einsum('hpwqc->hwpqc', img)
        img = np.mean(img, axis=(2,3,4))
        patch_min, patch_max = np.min(img), np.max(img)
        img = (img - patch_min) / (patch_max - patch_min + 10e-6)
        img_flatten = img.flatten()
        mask_preds = np.random.rand(img_flatten.shape[0])
        basket_dict = {}
        for i in range(img_flatten.shape[0]):
            id = math.floor(img_flatten[i] * self.num_tokens)
            if id not in basket_dict:
                basket_dict[id] = [i]
            else:
                basket_dict[id].append(i)
        for id, basket in basket_dict.items():
            token_id = random.sample(basket, 1)
            mask_preds[token_id] = -1.5
        return mask_preds

    def statistic_random_masking(self, sample):
        img = cv2.cvtColor(np.asarray(sample),cv2.COLOR_RGB2BGR)
        h = w = img.shape[1] // self.p
        img = img.reshape((h, self.p, w, self.p, 3))
        img = np.einsum('hpwqc->hwpqc', img)
        img = np.mean(img, axis=(2,3,4))
        patch_min, patch_max = np.min(img), np.max(img)
        img = (img - patch_min) / (patch_max - patch_min + 10e-6)
        img_flatten = img.flatten()
        mask_preds = np.random.rand(img_flatten.shape[0])
        basket_dict = {}
        for i in range(img_flatten.shape[0]):
            id = math.floor(img_flatten[i] * self.num_tokens)
            if id not in basket_dict:
                basket_dict[id] = [i]
            else:
                basket_dict[id].append(i)
        num_count = 0
        while num_count < self.num_tokens:
            for id in basket_dict.keys():
                if len(basket_dict[id]) > 0 and random.random() < 1 / self.num_tokens:
                    token_id = random.sample(basket_dict[id], 1)
                    mask_preds[token_id] = -1.5
                    num_count += 1
                    basket_dict[id].remove(token_id)
        return mask_preds

    def edge_masking(self, sample, target):
        img = cv2.cvtColor(np.asarray(sample),cv2.COLOR_RGB2GRAY)
        '''  canny '''
        # img = cv2.GaussianBlur(img, (3,3), 0)
        # img = cv2.Canny(img, 100, 200)
        ''' hog '''
        fd, img = feature.hog(img, orientations=9, pixels_per_cell=[8,8],
            cells_per_block=[2,2], visualize=True)

        h = w = img.shape[1] // self.p
        img = img.reshape((h, self.p, w, self.p, 1))
        img = np.einsum('hpwqc->hwpqc', img)
        img = np.mean(img, axis=(2,3,4))
        patch_min, patch_max = np.min(img), np.max(img)
        img = (img - patch_min) / (patch_max - patch_min + 10e-6)
        mask_preds = 1 - img.flatten()
        return mask_preds

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        mask_preds = self.statistic_masking(sample)
        # self.mask_visualizing(sample, target, mask_preds)
        
        sample = self.post_transform(sample)
        return sample, mask_preds, target

    def __len__(self) -> int:
        return len(self.samples)

class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")