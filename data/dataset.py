# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data

import numpy as np
from torchvision import transforms as T

import scipy.io as sio
import cv2
import sys

# coding:utf8
import os
import torch as t
from PIL import Image
from torch.utils import data

import numpy as np
from torchvision import transforms as T

import scipy.io as sio
from scipy.ndimage import rotate
import cv2
import sys

class NeedleDataset(data.Dataset):
    def  __init__(self, base_dir, split='train', num=None, matches_file=None, 
                 p_matrices_dir=None, use_augmented=False):
        self.split = split
        self.base_dir = base_dir
        self.p_matrices_dir = p_matrices_dir
        self.p_matrix_cache = {}
        self.use_augmented = use_augmented

        # P-matrix 매칭 정보 로드
        self.p_matrix_map = {}
        if matches_file is not None and p_matrices_dir is not None:
            self._load_p_matrix_mapping(matches_file)

        # unlabeled data는 별도 처리
        if split == 'unlabeled':
            self.image_dir = os.path.join(base_dir, 'unlabeled')
            self.image_list = sorted([f for f in os.listdir(self.image_dir)
                                      if f.endswith(('.png', '.jpg', '.jpeg'))])
            self.has_label = False
        
        else:
            # 증강된 폴더 또는 원본 폴더 선택
            if use_augmented and split in ['train', 'val', 'all', 'train2', 'val2', 'train3', 'train4', 'val4']:
                self.image_dir = os.path.join(base_dir, split, 'original_aug')
                self.label_dir = os.path.join(base_dir, split, 'label_aug')
            else:
                self.image_dir = os.path.join(base_dir, split, 'original')
                self.label_dir = os.path.join(base_dir, split, 'label')

            # Image List 생성
            self.image_list = sorted([f for f in os.listdir(self.image_dir)
                                      if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # num이 지정되면 labeled data만 선택 (원본 기준)
            if num is not None and split in ['train', 'all'] and use_augmented:
                # 원본 이미지만 선택 (rot가 없는 것)
                original_images = [f for f in self.image_list if 'rot' not in f]
                original_images = original_images[:num]
                
                # 선택된 원본에 대한 모든 증강 버전 포함
                selected_images = []
                for orig_img in original_images:
                    base_name = orig_img.split('.')[0]
                    # 해당 이미지의 모든 버전 찾기
                    selected_images.extend([f for f in self.image_list 
                                          if f.startswith(base_name)])
                
                self.image_list = sorted(selected_images)

            self.has_label = True

        print(f"Loaded {len(self.image_list)} images for split: {split}")
        if self.p_matrix_map:
            print(f"P-matrix mapping loaded for {len(self.p_matrix_map)} images.")

    def _load_p_matrix_mapping(self, matches_file):
        """
        top1_matches.txt 파일을 읽어 파싱하여 이미지명 -> P matrix 번호 매핑 생성
        """
        with open(matches_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    spatial_path = parts[1]

                    try:
                        spatial_num = int(spatial_path.split('/Spatials/')[1].split('/')[0])
                        self.p_matrix_map[img_name] = spatial_num
                    
                    except(IndexError, ValueError) as e:
                        print(f"Warning: Could not parse spatial number from path: {spatial_path}:{e}")
                        continue
    
    def _load_p_matrix(self, spatial_num, angle=0):
        """
        주어진 Spatial 번호와 회전 각도에 해당하는 P matrix 로드
        """
        cache_key = (spatial_num, angle)
        
        if cache_key in self.p_matrix_cache:
            return self.p_matrix_cache[cache_key]
        
        # P matrix 파일 경로 생성
        if angle == 0:
            mat_file = os.path.join(self.p_matrices_dir, f"{spatial_num}_P.mat")
        else:
            angle_str = f"rot{angle:+d}"
            mat_file = os.path.join(self.p_matrices_dir, f"{spatial_num}_P_{angle_str}.mat")
        
        if not os.path.exists(mat_file):
            print(f"Warning: P matrix file not found: {mat_file}")
            if angle != 0:
                return self._load_p_matrix(spatial_num, angle=0)
            return None
        
        try:
            mat_data = sio.loadmat(mat_file)
            p_data = {'P': mat_data['P'], 'P_T': mat_data['P_T']}
            self.p_matrix_cache[cache_key] = p_data
            return p_data
        except Exception as e:
            print(f"Error loading P matrix from {mat_file}: {e}")
            return None
    
    def _extract_angle_from_filename(self, filename):
        """
        파일명에서 회전 각도 추출
        예: '0001_rot-25.jpg' -> -25
            '0001.jpg' -> 0
        """
        if 'rot' not in filename:
            return 0
        
        try:
            # '_rot+25' 또는 '_rot-25' 형태에서 각도 추출
            angle_part = filename.split('rot')[1].split('.')[0]
            return int(angle_part)
        except:
            return 0

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        # 이미지 로드
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Grayscale로 로드
        image = Image.open(img_path).convert('L')
        image = np.array(image, dtype=np.float32) / 255.0

        # Label 로드
        if self.has_label:
            label_path = os.path.join(self.label_dir, img_name)
            label = Image.open(label_path).convert('L')
            label = np.array(label, dtype=np.float32) / 255.0
            label = (label > 0.5).astype(np.float32)
        else:
            label = np.zeros_like(image, dtype=np.float32)

        # (H, W) -> (1, H, W)
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)

        sample = {'image': image, 'label': label}

        # P matrix 추가
        if self.p_matrices_dir is not None:
            # 파일명에서 각도 추출
            angle = self._extract_angle_from_filename(img_name)
            
            # Unlabeled 데이터 처리
            if self.split == 'unlabeled':
                try:
                    # 파일명에서 spatial 번호 추출
                    # 예: '199_Spatial_031.jpg' -> 199
                    spatial_num = int(img_name.split('_')[0])
                    
                    p_data = self._load_p_matrix(spatial_num, angle=angle)
                    if p_data is not None:
                        sample['P'] = p_data['P']
                        sample['P_T'] = p_data['P_T']
                    else:
                        sample['P'] = None
                        sample['P_T'] = None
                except (IndexError, ValueError) as e:
                    print(f"Warning: Cannot extract spatial number from {img_name}: {e}")
                    sample['P'] = None
                    sample['P_T'] = None
            
            # Labeled 데이터 처리
            else:
                # 원본 이미지명 추출 (rot 부분 제거)
                if 'rot' in img_name:
                    base_img_name = img_name.split('_rot')[0] + '.' + img_name.split('.')[-1]
                else:
                    base_img_name = img_name
                
                if base_img_name in self.p_matrix_map:
                    spatial_num = self.p_matrix_map[base_img_name]
                    p_data = self._load_p_matrix(spatial_num, angle=angle)
                    if p_data is not None:
                        sample['P'] = p_data['P']
                        sample['P_T'] = p_data['P_T']
                    else:
                        sample['P'] = None
                        sample['P_T'] = None
                else:
                    print(f"Warning: {base_img_name} not found in p_matrix_map")
                    sample['P'] = None
                    sample['P_T'] = None
        else:
            sample['P'] = None
            sample['P_T'] = None

        return sample
        
