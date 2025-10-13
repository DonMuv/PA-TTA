"""
变化检测数据集
"""

import clip
import torch
import os
from PIL import Image
import numpy as np
import glob

from torch.utils import data

from .data_utils import CDDataAugmentation
import json


"""
CD data set with pixel-level labels；
├─image
├─image_post
├─label
└─list
"""
IMG_FOLDER_NAME = "A"
IMG_POST_FOLDER_NAME = 'B'
LIST_FOLDER_NAME = 'list'
ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix='.png' # jpg for gan dataset, others : png

# device = 'cpu'
# clip_B16, clip_preprocess = clip.load("ViT-B/16", device=device)

def generate_clip_heatmap(image: np.ndarray, text: str, patch_size=224, stride=64):
    # print("image.shape: ", image.shape)   # [16, 3, 256, 256]
    # image = image.permute(1, 2, 0)
    H, W, _ = image.shape 
    heatmap = np.zeros((H, W))
    count_map = np.zeros((H, W))

    text_token = clip.tokenize(text)
    with torch.no_grad():
        text_feat = clip_B16.encode_text(text_token)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch = Image.fromarray(image[y:y+patch_size, x:x+patch_size])
                patch_tensor = clip_preprocess(patch).unsqueeze(0)
                image_feat = clip_B16.encode_image(patch_tensor)
                image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
                sim = (image_feat @ text_feat.T).item()

                heatmap[y:y+patch_size, x:x+patch_size] += sim
                count_map[y:y+patch_size, x:x+patch_size] += 1

    heatmap = heatmap / (count_map + 1e-8)
    # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化
    return heatmap
    

def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.str_)
    if img_name_list.ndim == 2:
        return img_name_list[:, 0]
    return img_name_list


def load_image_label_list_from_npy(npy_path, img_name_list):
    cls_labels_dict = np.load(npy_path, allow_pickle=True).item()
    return [cls_labels_dict[img_name] for img_name in img_name_list]


def get_img_post_path(root_dir,img_name):
    return os.path.join(root_dir, IMG_POST_FOLDER_NAME, img_name)


def get_img_path(root_dir, img_name):
    return os.path.join(root_dir, IMG_FOLDER_NAME, img_name)


def get_label_path(root_dir, img_name):
    return os.path.join(root_dir, ANNOT_FOLDER_NAME, img_name.replace('.jpg', label_suffix))


def create_dict(path):
    with open(path, 'r', encoding='utf-8') as f:
        tx_list = json.load(f)

    tx_dict = {}
    for item in tx_list:
        path = item.get("image_path", "")
        key = path.split('/')[-1]  # 取文件名部分
        tx_dict[key] = item

    return tx_dict

            
def json_to_text_prompt(text, top_k=2, threshold=0.001):
    # categories = text
    categories = {k: float(v) for k, v in text.items() if k != "image_path"}
    # 按权重排序
    sorted_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)
    # 选权重大于阈值的top_k个
    filtered = [(cat, weight) for cat, weight in sorted_cats if weight >= threshold][:top_k]
    # 构造文本
    prompt_parts = [f"{cat} ({weight:.2f})" for cat, weight in filtered]
    prompt = "This image mainly contains " + ", ".join(prompt_parts) + "."
    if 'solar panel' in prompt:
        prompt_out = 'a photo of solar panels'
    else:
        prompt_out = 'a photo without solar panels'

    return prompt_out


class ImageDataset(data.Dataset):
    """VOCdataloder"""
    def __init__(self, root_dir, split='train', img_size=256, is_train=True,to_tensor=True):
        super(ImageDataset, self).__init__()
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.split = split  #train | train_aug | val
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        
        # self.list_path = os.path.join(self.root_dir, LIST_FOLDER_NAME, self.split+'.txt')
        self.img_name_list = os.listdir(os.path.join(self.root_dir, 'A'))
        # self.img_name_list = list(map(lambda s: self.split + '/' + s, self.img_list))

        self.A_size = len(self.img_name_list)  # get the size of dataset A
        self.to_tensor = to_tensor
        if is_train:
            self.augm = CDDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=True,
                random_color_tf=True
            )
        else:
            self.augm = CDDataAugmentation(
                img_size=self.img_size
            )
    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img, img_B], _ = self.augm.transform([img, img_B],[], to_tensor=self.to_tensor)

        return {'A': img, 'B': img_B, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class CDDataset(ImageDataset):

    def __init__(self, root_dir, img_size, split='train', is_train=True, to_tensor=True, loader=None):
        super(CDDataset, self).__init__(root_dir, img_size=img_size, split=split, is_train=is_train,
                                        to_tensor=to_tensor)
        # self.tx_A_dict = create_dict(os.path.join(self.root_dir, f'{split}_A_clipcls.json'))
        # self.tx_B_dict = create_dict(os.path.join(self.root_dir, f'{split}_B_clipcls.json'))
        # self.loader = loader

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.root_dir, self.img_name_list[index % self.A_size])
        B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        img_B = np.asarray(Image.open(B_path).convert('RGB'))

        if self.split == 'val':
            L_path = get_label_path(self.root_dir, self.img_name_list[index % self.A_size])
            label = np.array(Image.open(L_path), dtype=np.uint8)

            # txA_item = self.tx_A_dict.get(name, None)
            # txB_item = self.tx_B_dict.get(name, None)
            # txA = json_to_text_prompt(txA_item, top_k=2, threshold=0.001)
            # txB = json_to_text_prompt(txB_item, top_k=2, threshold=0.001)
            
        else:
            # txA_item = self.tx_A_dict.get(name, None)
            # txB_item = self.tx_B_dict.get(name, None)
            # txA = json_to_text_prompt(txA_item, top_k=2, threshold=0.001)
            # txB = json_to_text_prompt(txB_item, top_k=2, threshold=0.001)
            label = np.zeros((self.img_size, self.img_size))

        # clip_heatA = generate_clip_heatmap(img, txA)
        # clip_heatB = generate_clip_heatmap(img_B, txB)
            
        [img, img_B], [label] = self.augm.transform([img, img_B], [label], to_tensor=self.to_tensor)
        # regionA = self.loader(A_path)
        # regionB = self.loader(B_path)
        sam_mask_pathA = os.path.join('../../unsuper/PVP-Germany/sam_geo_outA', name[:-3] + 'tif')
        sam_mask_pathB = os.path.join('../../unsuper/PVP-Germany/sam_geo_outB', name[:-3] + 'tif')
        # sam_mask_pathA = os.path.join('temp_img_out/sam_geo_outA', name[:-3] + 'tif')
        # sam_mask_pathB = os.path.join('temp_img_out/sam_geo_outB', name[:-3] + 'tif')
        sam_maskA = np.asarray(Image.open(sam_mask_pathA))
        sam_maskB = np.asarray(Image.open(sam_mask_pathB))
        sam_maskA = sam_maskA / 255
        sam_maskB = sam_maskB / 255
        sam_mask = np.maximum(sam_maskB - sam_maskA, 0)
        
        # print(segmentation_masks.shape)
        
        if self.split == 'val':
            # return {'A': img, 'B': img_B, 'L': label}
            return {'name': name, 'A': img, 'B': img_B, 'L': label, 'sam_mask': sam_mask}
            # return {'name': name, 'A': img, 'B': img_B, 'L': label, 'Atx': txA, 'Btx': txB, 'regionA': regionA, 'regionB': regionB}
            
        return {'name': name, 'A': img, 'B': img_B, 'Atx': txA, 'Btx': txB}

