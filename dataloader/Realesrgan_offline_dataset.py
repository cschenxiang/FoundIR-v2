import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
import torch.nn.functional as F
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp, img2tensor, tensor2img
from basicsr.utils.img_process_util import filter2D
from PIL import Image
import json
from transformers import CLIPImageProcessor
from torch import nn
from torchvision import transforms
from torch.utils import data as data
from torchvision.transforms.functional import normalize
from .realesrgan import RealESRGAN_degradation
import cv2
import random
from glob import glob
from collections import OrderedDict
import yaml
from PIL import Image
import os
from glob import glob
import numpy as np
from collections import deque
import multiprocessing as mp

def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def opt_parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)

    return opt


def convert_image_to_fn(img_type, image, minsize=512, eps=0.02):
    width, height = image.size
    if min(width, height) < minsize:
        scale = minsize / min(width, height) + eps
        image = image.resize((math.ceil(width * scale), math.ceil(height * scale)))

    if image.mode != img_type:
        return image.convert(img_type)
    return image


def exists(x):
    return x is not None

def build_proportional_batches(hq_root_dir, batch_size=16):
    subfolders = [os.path.join(hq_root_dir, d) for d in os.listdir(hq_root_dir)
                  if os.path.isdir(os.path.join(hq_root_dir, d))]

    folder_to_images = {}
    total_images = 0
    for folder in subfolders:
        image_paths = sorted(glob(os.path.join(folder, '**', '*.*'), recursive=True))
        image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        folder_to_images[folder] = deque(image_paths)
        total_images += len(image_paths)

    folder_to_ratio = {folder: len(imgs) / total_images for folder, imgs in folder_to_images.items()}

    final_img_list = []
    total_batches = int(np.ceil(total_images / batch_size))
    folder_list = list(folder_to_ratio.keys())
    num_folders = len(folder_list)
    round_robin_index = 0

    for _ in range(total_batches):
        batch = []

        alloc_per_class = {folder: int(round(batch_size * folder_to_ratio[folder])) for folder in folder_list}

        current_total = sum(alloc_per_class.values())
        diff = batch_size - current_total

        if diff != 0:
            for i in range(abs(diff)):
                idx = (round_robin_index + i) % num_folders
                folder = folder_list[idx]
                alloc_per_class[folder] += 1 if diff > 0 else -1

        round_robin_index = (round_robin_index + abs(diff)) % num_folders

        for folder, count in alloc_per_class.items():
            for _ in range(count):
                if folder_to_images[folder]:
                    batch.append(folder_to_images[folder].popleft())

        final_img_list.extend(batch)

    return final_img_list


def process(img_file):
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.gif', '.tiff', '.webp'}

    folder_lists = {}

    top_categories = [d for d in os.listdir(img_file) if os.path.isdir(os.path.join(img_file, d))]

    for cat in top_categories:
        cat_path = os.path.join(img_file, cat)
        all_images = []

        for root, _, files in os.walk(cat_path):
            for f in files:
                if os.path.splitext(f)[1] in image_extensions:
                    rel_img_path = os.path.relpath(os.path.join(root, f), img_file)
                    all_images.append(rel_img_path)

        if all_images:
            folder_lists[cat] = all_images

    folder_names = list(folder_lists.keys())
    total_count = sum(len(imgs) for imgs in folder_lists.values())

    ratios = [len(folder_lists[name]) / total_count for name in folder_names]
    folder_np_dict = {name: np.array(paths) for name, paths in folder_lists.items()}

    return total_count, ratios, folder_np_dict, folder_names

_GLOBAL_COUNTER = mp.Value("i", 0)
_GLOBAL_LOCK = mp.Lock()


def get_curr_prob(index_, schedule, data_types):
    for stage in schedule:
        if index_ < stage["limit"]:
            stage_prob = stage["prob"]
            probs = [stage_prob.get(t, 0.0) for t in data_types]

            s = sum(probs)
            if s == 0: return [1.0 / len(data_types)] * len(data_types)
            return [p / s for p in probs]

SAMPLE_SCHEDULE = [
    # index_ < 300000
    {"limit": 300000, "prob": {"Task1":0.2, "Task2":0.2, "Task3":0.2, "Task4":0.2, "Task5":0.2}},
    # 300000 <= index_ < 600000
    {"limit": 600000, "prob": {"Task1":0.3, "Task2":0.1, "Task3":0.2, "Task4":0.1, "Task5":0.3}},
    # 600000 <= index_ < 900000
    {"limit": 900000, "prob": {"Task1":0.3, "Task2":0.1, "Task3":0.1, "Task4":0.1, "Task5":0.4}},
    # >= 900000
    {"limit": float("inf"), "prob": {"Task1":0.2, "Task2":0.1, "Task3":0.1, "Task4":0.1, "Task5":0.5}},
]

class LocalImageDataset(data.Dataset):
    def __init__(self,
                 img_file=None,
                 yml_kernel=None,
                 image_size=512,
                 tokenizer=None,
                 tokenizer_2=None,
                 center_crop=False,
                 random_flip=True,
                 resize_bak=True,
                 convert_image_to="RGB",
                 t_drop_rate=0.05,
                 ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2

        self.resize_bak = resize_bak

        self.crop_size = image_size

        self.t_drop_rate = t_drop_rate

        total_count, ratios, folder_np_dict, folder_names = process(img_file)

        self.total_count = total_count
        self.data_types = folder_names
        self.data_collection = folder_np_dict
        self.img_file = img_file



    def __getitem__(self, index):
        with _GLOBAL_LOCK:
            index_ = _GLOBAL_COUNTER.value
            _GLOBAL_COUNTER.value += 1

        cur_prob = get_curr_prob(index_, SAMPLE_SCHEDULE, self.data_types)
        data_type = random.choices(self.data_types, cur_prob)[0]

        index = np.random.randint(len(self.data_collection[data_type]))
        img_path = self.data_collection[data_type][index]
        img_path = os.path.join(self.img_file, img_path)
        json_base = img_path.replace("HQ_folder", "json_folder")
        json_path = os.path.splitext(json_base)[0] + ".json"
        lq_img_path = img_path.replace("HQ_folder", "LQ_folder")

        gt_path = img_path
        data = json.load(open(json_path))
        init_text = data["caption"]
        words = init_text.split()
        words = words[3:]
        words[0] = words[0].capitalize()
        text = ' '.join(words)
        text = text.split('. ')
        text = '. '.join(text[:2]) + '.'

        image = Image.open(img_path).convert('RGB')

        try:
            lq_image = Image.open(lq_img_path).convert('RGB')
        except FileNotFoundError:
            alt_ext = '.png' if lq_img_path.lower().endswith('.jpg') else '.jpg'
            lq_img_path = os.path.splitext(lq_img_path)[0] + alt_ext
            lq_image = Image.open(lq_img_path).convert('RGB')

        if 'Task5' in img_path:
            mode = random.choice([Image.NEAREST, Image.BILINEAR, Image.BICUBIC])
            lr_w, lr_h = lq_image.size
            lq_image = lq_image.resize((lr_w * 4, lr_h * 4), mode)

        w, h = lq_image.size
        pil_img = np.array(image)
        pil_lr_img = np.array(lq_image)
        pil_img, pil_lr_img = augment([pil_img, pil_lr_img], hflip=True, rotation=False)

        crop_pad_size = self.crop_size // 1

        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            pil_lr_img = cv2.copyMakeBorder(pil_lr_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

            pad_h = max(0, self.crop_size - h * 1)
            pad_w = max(0, self.crop_size - w * 1)
            pil_img = cv2.copyMakeBorder(pil_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if pil_lr_img.shape[0] > crop_pad_size or pil_lr_img.shape[1] > crop_pad_size:
            h, w = pil_lr_img.shape[0:2]
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)

            pil_lr_img = pil_lr_img[top: top + crop_pad_size, left: left + crop_pad_size, ...]
            pil_img = pil_img[top * 1: (top + crop_pad_size) * 1, left * 1: (left + crop_pad_size) * 1, ...]

        else:
            top = 0
            left = 0

        lq_image = Image.fromarray(pil_lr_img)


        image = Image.fromarray(pil_img)
        original_size = torch.tensor([h * 1, w * 1])
        crop_coords_top_left = torch.tensor([top * 1, left * 1])

        GT_image_t = np.asarray(image) / 255.
        LR_image_t = np.asarray(lq_image) / 255.

        GT_image_t, LR_image_t = img2tensor([GT_image_t, LR_image_t], bgr2rgb=False, float32=True)
        LR_image_t = LR_image_t * 2.0 - 1.0
        GT_image_t = GT_image_t * 2.0 - 1.0

        rand_num = random.random()
        if rand_num < self.t_drop_rate:
            text = ""

        text_input_ids = self.tokenizer(
            text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        text_input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids = self.tokenizer(
            '',
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        null_text_input_ids_2 = self.tokenizer_2(
            '',
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            'lq_image': LR_image_t,
            "image": GT_image_t,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "original_size": original_size,
            "crop_coords_top_left": crop_coords_top_left,
            "target_size": torch.tensor([crop_pad_size, crop_pad_size]),
            'gt_path': gt_path,
            'null_text_input_ids': null_text_input_ids,
            "null_text_input_ids_2": null_text_input_ids_2
        }

    def __len__(self):
        return self.total_count