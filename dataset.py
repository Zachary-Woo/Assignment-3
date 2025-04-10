import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from torchvision import transforms
import glob
from skimage import io
import random

class NuInsSegDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train', train_val_test_split=[0.7, 0.1, 0.2], seed=42):
        """
        Args:
            root_dir (string): Directory with the NuInsSeg dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
            mode (string): 'train', 'val', or 'test'
            train_val_test_split (list): Ratios for train/val/test split
            seed (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # Find all tissue subfolders
        self.tissue_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        # Create a list of (image_path, mask_path) tuples
        self.image_mask_pairs = []
        for tissue in self.tissue_folders:
            img_dir = os.path.join(root_dir, tissue, 'tissue images')
            mask_dir = os.path.join(root_dir, tissue, 'label masks')
            
            if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
                continue
                
            img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            
            for img_file in img_files:
                img_path = os.path.join(img_dir, img_file)
                mask_file = img_file.replace('.png', '.tif')
                mask_path = os.path.join(mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    self.image_mask_pairs.append((img_path, mask_path))
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Shuffle and split the dataset
        random.shuffle(self.image_mask_pairs)
        total_samples = len(self.image_mask_pairs)
        train_size = int(train_val_test_split[0] * total_samples)
        val_size = int(train_val_test_split[1] * total_samples)
        
        if mode == 'train':
            self.image_mask_pairs = self.image_mask_pairs[:train_size]
        elif mode == 'val':
            self.image_mask_pairs = self.image_mask_pairs[train_size:train_size+val_size]
        elif mode == 'test':
            self.image_mask_pairs = self.image_mask_pairs[train_size+val_size:]
        
        print(f"Mode: {mode}, Total samples: {len(self.image_mask_pairs)}")
    
    def __len__(self):
        return len(self.image_mask_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        
        # Load RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (instance segmentation)
        mask = io.imread(mask_path)
        
        # Ensure mask is in the correct format (uint16 for instance segmentation)
        if mask.dtype != np.uint16:
            mask = mask.astype(np.uint16)
        
        # Apply transformations
        if self.transform:
            # Create a sample dictionary
            sample = {'image': image, 'mask': mask}
            sample = self.transform(sample)
            image, mask = sample['image'], sample['mask']
        else:
            # Convert to tensor
            image = torch.from_numpy(image.transpose((2, 0, 1))).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': img_path,
            'mask_path': mask_path
        }


class Resize:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Resize image
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        
        # Resize mask (nearest neighbor to preserve instance IDs)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        return {'image': image, 'mask': mask}


class RandomCrop:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        
        if h > self.size and w > self.size:
            i = random.randint(0, h - self.size)
            j = random.randint(0, w - self.size)
            
            image = image[i:i+self.size, j:j+self.size]
            mask = mask[i:i+self.size, j:j+self.size]
        
        return {'image': image, 'mask': mask}


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.p:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        
        return {'image': image, 'mask': mask}


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() < self.p:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        return {'image': image, 'mask': mask}


class RandomRotation:
    def __init__(self, degrees=10):
        self.degrees = degrees
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        angle = random.uniform(-self.degrees, self.degrees)
        h, w = image.shape[:2]
        center = (w/2, h/2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        return {'image': image, 'mask': mask}


class Normalize:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        return {'image': image, 'mask': mask}


class ToTensor:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        # Convert to tensor
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        
        return {'image': image, 'mask': mask}


def get_transform(mode='train', image_size=1024):
    transforms_list = []
    
    # Resize to target size
    transforms_list.append(Resize(image_size))
    
    if mode == 'train':
        # Data augmentation only for training
        transforms_list.extend([
            RandomCrop(image_size),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation(10),
        ])
    
    # Add normalization and tensor conversion
    transforms_list.extend([
        Normalize(),
        ToTensor()
    ])
    
    # Create a composed transform
    def composed_transforms(sample):
        for transform in transforms_list:
            sample = transform(sample)
        return sample
    
    return composed_transforms


def create_dataloaders(root_dir, batch_size=4, image_size=1024, num_workers=4):
    """
    Create dataloaders for training, validation, and testing
    """
    train_dataset = NuInsSegDataset(
        root_dir=root_dir,
        transform=get_transform(mode='train', image_size=image_size),
        mode='train'
    )
    
    val_dataset = NuInsSegDataset(
        root_dir=root_dir,
        transform=get_transform(mode='val', image_size=image_size),
        mode='val'
    )
    
    test_dataset = NuInsSegDataset(
        root_dir=root_dir,
        transform=get_transform(mode='test', image_size=image_size),
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 