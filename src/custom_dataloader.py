from abc import abstractmethod
from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, transform=None,
                image_idx=None, class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        self.root = root
        self.filepaths = filepaths
        self.labels = labels

        self.transform = transform
        self.image_idx = image_idx
        self.class_idx = class_idx
    
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError('Subclasses need to implement it!')

    def __len__(self):
        return len(self.filepaths)
    
    
class aPYDataset(CustomImageDataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, transform=None,
                image_idx=None, class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        
        super().__init__(root, filepaths, labels,
                        transform, image_idx, class_idx)

    def __getitem__(self, index):
        path = self.filepaths[index]
        im = Image.open(f'{self.root}{self.filepaths[index]}').convert('RGB')
        img = im

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            idx = self.image_idx[path]
            cls = self.labels[idx]
            cls_idx = self.class_idx[cls]
            label = torch.tensor(cls_idx)
            return img, label, idx
        else:
            return img, None, idx
        
        
class LADDataset(CustomImageDataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, transform=None,
                image_idx=None, class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        
        super().__init__(root, filepaths, labels,
                        transform, image_idx, class_idx)

    def __getitem__(self, index):
        path = self.filepaths[index]
        im = Image.open(f'{self.root}{self.filepaths[index]}').convert('RGB')
        img = im

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            idx = self.image_idx[path]
            cls = self.labels[idx]
            cls_idx = self.class_idx[cls]
            label = torch.tensor(cls_idx)
            return img, label, idx
        else:
            return img, None, idx
        
class SUNDataset(CustomImageDataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, transform=None,
                image_idx=None, class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        
        super().__init__(root, filepaths, labels,
                        transform, image_idx, class_idx)

    def __getitem__(self, index):
        path = self.filepaths[index]
        im = Image.open(f'{self.root}{self.filepaths[index]}').convert('RGB')
        img = im

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            idx = self.image_idx[path]
            cls = self.labels[idx]
            cls_idx = self.class_idx[cls]
            label = torch.tensor(cls_idx)
            return img, label, idx
        else:
            return img, None, idx
        
        
class CUBDataset(CustomImageDataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, 
                 transform=None, image_idx=None, 
                 class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        
        super().__init__(root, filepaths, labels,
                        transform, image_idx, class_idx)

    def __getitem__(self, index):
        path = self.filepaths[index]
        im = Image.open(f'{self.root}{self.filepaths[index]}').convert('RGB')
        img = im

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            idx = self.image_idx[path]
            cls = self.labels[idx]
            cls_idx = self.class_idx[cls]
            label = torch.tensor(cls_idx)
            return img, label, idx
        else:
            return img, None, idx
        
class AwADataset(CustomImageDataset):
    """
    A custom dataset used to create dataloaders.
    """
    def __init__(self, root, filepaths, labels=None, 
                 transform=None, image_idx=None, 
                 class_idx=None):
        """
        Create a new CustomImageDataset.
        
        :param filepaths: A list of filepaths. 
        :param labels: A list of labels
        :param label_map: A dictionary to map string labels to intergers
        :param transform: A transform to perform on the images
        """
        
        super().__init__(root, filepaths, labels,
                        transform, image_idx, class_idx)

    def __getitem__(self, index):
        path = self.filepaths[index]
        im = Image.open(f'{self.root}{self.filepaths[index]}').convert('RGB')
        img = im

        if self.transform is not None:
            img = self.transform(img)

        if self.labels is not None:
            idx = self.image_idx[path]
            cls = self.labels[idx]
            cls_idx = self.class_idx[cls]
            label = torch.tensor(cls_idx)
            return img, label, idx
        else:
            return img, None, idx