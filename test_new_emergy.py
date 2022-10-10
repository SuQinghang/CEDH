import os
import torch
import numpy as np

from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from cedh import generate_code
from data.transform import train_transform, query_transform, Onehot, encode_onehot
from PIL import Image

from utils.evaluate import mean_average_precision

class ImagenetDataset(Dataset):
    classes = None
    class_to_idx = None

    def __init__(self, root, num_classes, isnew=False, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []

        # Assume file alphabet order is the class order
        if ImagenetDataset.class_to_idx is None:
            ImagenetDataset.classes, ImagenetDataset.class_to_idx = self._find_classes(root)

        for i, cl in enumerate(ImagenetDataset.classes):
            cur_class = os.path.join(self.root, cl)
            files = os.listdir(cur_class)
            files = [os.path.join(cur_class, i) for i in files]
            num_old = 4 * (len(files)//5)
            if isnew:
                files = files[num_old:]
            else:
                files = files[:num_old]
            self.data.extend(files)
            self.targets.extend([ImagenetDataset.class_to_idx[cl] for i in range(len(files))])
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        # self.onehot_targets = torch.from_numpy(encode_onehot(self.targets, 10)).float()
        self.onehot_targets = encode_onehot(self.targets, num_classes)
   
    def get_onehot_targets(self):
        # return self.onehot_targets
        return torch.from_numpy(self.onehot_targets).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        img, target = self.data[item], self.targets[item]

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, item

    def _find_classes(self, dir):
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
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx



device = torch.device('cuda:2')
# load dataset
cifar_database = '/data2/suqinghang/Dataset/cifar-10/retrieval'
imagenet_database = '/data2/suqinghang/Dataset/Imagenet100/database'
bs = 128
old_dataset = ImagenetDataset(cifar_database, 10, isnew=False, transform=query_transform(), target_transform=Onehot())
old_dataloader = DataLoader(old_dataset, batch_size=bs, num_workers=8, shuffle=False, pin_memory=True)
new_dataset = ImagenetDataset(cifar_database, 10, isnew=True, transform=query_transform(), target_transform=Onehot())
new_dataloader = DataLoader(new_dataset, batch_size=bs, num_workers=8, shuffle=False, pin_memory=True)

original_length = 60
target_length = original_length + 4
# load adsh model
adsh_path = 'checkpoints/adsh/cifar-10/cifar-10_60bits_topk@None-2022-07-05-14:43'
adsh_model = torch.load(os.path.join(adsh_path, 'model-{}.t'.format(original_length))).to(device)
# load cedh model
cedh_path = 'checkpoints/cedh/cifar-10/cifar-10_64bits_topk@None-2022-07-08-11:37'
cedh_model = torch.load(os.path.join(cedh_path, 'model-{}.t'.format(target_length))).to(device)
cedh_W = torch.load(os.path.join(cedh_path, 'W-{}.t').format(target_length)).to(device)

# 已有的由adsh生成的k bits
old_code_k = generate_code(adsh_model, old_dataloader, original_length, device).to(device) 
old_targets = old_dataset.get_onehot_targets().to(device)

# 由CEDH的W生成新的c bits,和已有的k bits构成old data的k' bits
old_code_c = (old_code_k @ cedh_W).sign()
old_code_k_prime = torch.cat((old_code_k, old_code_c), 1)

# 由CEDH的cnn生成new data的k' bits
new_code_k_prime = generate_code(cedh_model, new_dataloader, target_length, device).to(device)
new_targets = new_dataset.get_onehot_targets().to(device)

# 由CEDH的cnn生成query data的k' bits
query_code_k_prime = torch.load(os.path.join(cedh_path, 'query_code{}.t').format(target_length)).to(device)
query_targets = torch.load(os.path.join(cedh_path, 'query_targets{}.t').format(target_length)).to(device)


overall_code = torch.vstack((old_code_k_prime, new_code_k_prime))
overall_targets = torch.vstack((old_targets, new_targets))

# 分别测试k和k' bits, CEDH的效果
k_map = mean_average_precision(query_code=query_code_k_prime[:, :original_length],
                                database_code=overall_code[:, :original_length],
                                query_labels=query_targets,
                                database_labels=overall_targets,
                                device=device,
                                topk=None)
map = mean_average_precision(query_code=query_code_k_prime,
                                database_code=overall_code,
                                query_labels=query_targets,
                                database_labels=overall_targets,
                                device=device,
                                topk=None)
print('k-map:{:.4f}| map:{:.4f}'.format(k_map, map))

