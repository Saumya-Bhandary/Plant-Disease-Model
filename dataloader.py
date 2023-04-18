import os
import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(self.root_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.images = []
        for c in self.classes:
            images_dir = os.path.join(self.root_dir, c)
            for image in os.listdir(images_dir):
                self.images.append((os.path.join(images_dir, image), self.class_to_idx[c]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, label = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

def get_dataloader(root_dir, batch_size, shuffle=True, num_workers=4):
    dataset = CustomDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader
