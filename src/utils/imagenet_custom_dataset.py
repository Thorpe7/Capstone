''' Custom dataset for imagenet val & test sets. '''

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    def __init__(self, img_dir:str, labels_file:str, solution_path: str, transform=None):
        """ Creates custom dataset object for functionality w/ pytorch.
        Args:
            img_dir (string): Directory with all the images
            labels_file (string): Path to the CSV file with annotations.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transform
        self.solution_path = pd.read_csv(solution_path)
        with open(labels_file,'r') as fp:
            lines = fp.readlines()

        self.labels_in_order = [line.strip().split(" ")[0] for line in lines]
    
    def __len__(self):
        return len(self.solution_path)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.solution_path.iloc[idx, 0])+".JPEG"
        image = Image.open(img_name).convert('RGB')
        full_string = self.solution_path.iloc[idx, 1]
        label = full_string.split(" ")[0]
        label_int = self.labels_in_order.index(label)
        

        if self.transform:
            image = self.transform(image)
        return image, label_int
