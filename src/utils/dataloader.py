import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class ISIC2019Dataset(Dataset):
    def __init__(self, img_dir, gt_csv_path, meta_csv_path, transform=None):
        """
        Custom PyTorch Dataset for ISIC 2019 with Multi-Modal Metadata.
        Now includes Age, Sex, and 8 Anatomical Sites.
        """
        self.img_dir = img_dir
        self.transform = transform

        # 1. Load the CSV files
        print("Loading and merging CSV files...")
        gt_df = pd.read_csv(gt_csv_path)
        meta_df = pd.read_csv(meta_csv_path)

        # 2. Merge Ground Truth and Metadata on the 'image' column
        self.df = pd.merge(gt_df, meta_df, on='image', how='inner')

        # 3. Extract Image Names
        self.image_names = self.df['image'].values

        # 4. Process Labels (Convert from One-Hot to Class Indices 0-7)
        self.class_columns = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC']
        self.labels = self.df[self.class_columns].idxmax(axis=1).map(
            {col: idx for idx, col in enumerate(self.class_columns)}
        ).values

        # 5. Process Metadata (Age, Sex, Anatomical Site)
        self.process_metadata()

    def process_metadata(self):
        """
        Applies Neutral Imputation for Age/Sex, and One-Hot Encoding for Anatomical Site.
        """
        # Process Age (Index 0)
        ages = self.df['age_approx'].values / 100.0
        ages = np.nan_to_num(ages, nan=0.5)

        # Process Sex (Index 1)
        sexes = self.df['sex'].map({'male': 1.0, 'female': 0.0}).values
        sexes = np.nan_to_num(sexes, nan=0.5)

        # Process Anatomical Site (Indices 2 through 9)
        # The 8 standard anatomical sites in ISIC 2019
        sites = [
            'anterior torso', 'head/neck', 'lateral torso',
            'lower extremity', 'oral/genital', 'palms/soles',
            'posterior torso', 'upper extremity'
        ]

        # Create a one-hot matrix of shape (N, 8)
        site_matrix = np.zeros((len(self.df), len(sites)))

        for i, site in enumerate(sites):
            # Set to 1.0 if it matches the site, otherwise it remains 0.0
            site_matrix[:, i] = (self.df['anatom_site_general'] == site).astype(float)

        # Stack Age (1) + Sex (1) + Sites (8) into a single metadata array: Shape (N, 10)
        self.metadata = np.column_stack((ages, sexes, site_matrix))
        self.metadata = torch.tensor(self.metadata, dtype=torch.float32)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        # 1. Get Image Path
        img_name = self.image_names[idx] + ".jpg"
        img_path = os.path.join(self.img_dir, img_name)

        # 2. Load Image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {img_path}")
            raise e

        # 3. Apply Transforms
        if self.transform:
            image = self.transform(image)

        # 4. Get Metadata and Label
        meta = self.metadata[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, meta, label


def get_dataloaders(img_dir, gt_csv, meta_csv, batch_size=32):
    """
    Returns the training dataloader.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ISIC2019Dataset(
        img_dir=img_dir,
        gt_csv_path=gt_csv,
        meta_csv_path=meta_csv,
        transform=transform
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return dataloader, dataset