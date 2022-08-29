from torchvision import transforms
import numpy as np
from typing import Any, Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR100

class MINICIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        mini_label_names: list,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        self.mini_label_names = mini_label_names
        self.mini_data: Any = []
        self.mini_targets = []
        all_mini_idx, mini_labels = self.get_all_mini_idx(mini_label_names, self.class_to_idx, self.targets)
        for idx in all_mini_idx:
            self.mini_data.append(self.data[idx])
            self.mini_targets.append(mini_labels.index(self.targets[idx]))

        self.data = self.mini_data
        self.targets = self.mini_targets
    #     self._load_meta()

    # def _load_meta(self,mini_label_names) -> None:
        self.class_to_idx = {_class: i for i, _class in enumerate(self.mini_label_names)}

    def get_all_mini_idx(self, mini_label_names, class_idxs: list, label: list):
        mini_labels = []
        for mini_label_name in mini_label_names:
            for class_name in class_idxs.keys():
                if mini_label_name in class_name:
                    mini_labels.append(class_idxs[class_name])
                    break
        label = np.array(label)
        all_mini_labels = []
        for i in mini_labels:
            all_mini_labels.extend(np.argwhere(label == i).flatten().tolist())
        return all_mini_labels,mini_labels

def get_dataset(arg):
    assert arg.mode in ['train', 'test']
    if arg.mode == 'train':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  # 数据增强
            transforms.ToTensor(),
            transforms.Normalize(arg.mean, arg.std)])
        dataset = MINICIFAR100(root=arg.data_path, mini_label_names=arg.mini_label_names, train=True, transform=transform, download=True)
    else:
        arg.mode = 'test'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(arg.mean, arg.std)])
        dataset = MINICIFAR100(root=arg.data_path, mini_label_names=arg.mini_label_names, train=False, transform=transform, download=True)

    return dataset

def get_dataloader(dataset, arg):
    if arg.mode == 'train':
        return DataLoader(dataset, arg.batch_size, shuffle=True, num_workers=arg.num_workers)
    else:
        return DataLoader(dataset, arg.batch_size, shuffle=False, num_workers=arg.num_workers)
