import torch
import os
from torchvision import transforms
from PIL import Image


class CUBDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path, mode, transforms_mode='test', nb_train_images=5864, nb_test_images=5924):

        self.ds_path = ds_path
        self.mode = mode
        self.transforms_mode = transforms_mode
        self.paths, self.labels, self.ids = [], [], []

        self.sz_crop = 224
        self.sz_resize = 256
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        with open(os.path.join(self.ds_path, 'images.txt'), 'r') as f:
            images = f.readlines()

        with open(os.path.join(self.ds_path, 'image_class_labels.txt'), 'r') as f:
            image_class_labels = f.readlines()

        for labels, paths in zip(image_class_labels, images):
            id1, label = labels.replace('\n', '').split()
            id2, path = paths.replace('\n', '').split()
            assert int(id1) == int(id2)

            # using first 100 classes for training
            if mode is 'train' and int(label) < 101:
                self.labels.append(int(label))
                self.paths.append(path)
                self.ids.append(int(id1) - 1)

            # and second 100 classes for testing
            elif mode is 'test' and int(label) >= 101:
                self.labels.append(int(label))
                self.paths.append(path)
                self.ids.append(int(id1) - 1)

        if mode is 'train':
            assert len(self.labels) == nb_train_images
        else:
            assert len(self.labels) == nb_test_images

    def apply_augmentation(self, img):
        if self.transforms_mode == 'train':
            transforms_ = transforms.Compose([
                transforms.RandomResizedCrop(self.sz_crop),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            ])
        else:
            transforms_ = transforms.Compose([
                transforms.Resize(self.sz_resize),
                transforms.CenterCrop(self.sz_crop),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                )
            ])
        return transforms_(img)

    def __len__(self):
        return len(self.labels)

    def get_nb_classes(self):
        return len(set(self.labels))

    def unique_labels(self):
        return set(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.ds_path + '/images/' + self.paths[idx])
        if len(list(img.split())) != 3:
            img = img.convert('RGB')
        img = self.apply_augmentation(img)
        return img, self.labels[idx], idx

    def get_label(self, idx):
        return self.labels[idx]

    def get_within_indexes(self, idxs):
        self.paths = [self.paths[i] for i in idxs]
        self.labels = [self.labels[i] for i in idxs]
        self.ids = [self.ids[i] for i in idxs]
