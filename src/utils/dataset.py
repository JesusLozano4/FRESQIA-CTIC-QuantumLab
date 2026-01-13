import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import random

class Two_Dataset:
    def __init__(self, root="../dataset/hojas2clases", image_size=64, batch_size=256,
                 test_size=0.2, random_state=42, examples_per_class=10000,
                 allowed_classes=None, output='np', dset="Colors"):
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self.examples_per_class = examples_per_class
        self.output = output


        if dset == "2hojas":
            self.class_dict = {
                'healthy': 0, 'scorch': 1
            }
            
        self.allowed_classes = [self.class_dict[c] for c in allowed_classes] if allowed_classes else None

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_loaders(self):
        dataset = datasets.ImageFolder(root=self.root, transform=self.transform)

        if self.allowed_classes:
            indices = [i for i, (_, target) in enumerate(dataset.samples) if target in self.allowed_classes]
            dataset = Subset(dataset, indices)
            # NUEVO: Mostrar cuántas imágenes hay por clase después de filtrar
            print("Images after applying 'allowed_classes':")
            print(self.count_images_per_class(dataset))

        if self.examples_per_class > 0:
            indices = self._limit_examples_per_class(dataset)
            dataset = Subset(dataset.dataset if isinstance(dataset, Subset) else dataset, indices)

        train_indices, val_indices = self._split_dataset(dataset)

        
        train_set = Subset(dataset, train_indices)
        val_set = Subset(dataset, val_indices)

        index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.allowed_classes)} if self.allowed_classes else None

        if index_mapping:
            train_set = self._remap_subset_labels(train_set, index_mapping)
            val_set = self._remap_subset_labels(val_set, index_mapping)

        g = torch.Generator(device="cpu").manual_seed(self.random_state)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, generator=g)

        if self.output == "dl":
            return train_loader, val_loader

        elif self.output == "np":
            X_train, y_train = self._loader_to_numpy(train_loader)
            X_val, y_val = self._loader_to_numpy(val_loader)
            return X_train, y_train, X_val, y_val, index_mapping
        else:
            raise ValueError(f"Unsupported output format: {self.output}. Use 'dl' or 'np'.")

    def _remap_subset_labels(self, subset, index_mapping):
        class RemappedSubset(Dataset):
            def __init__(self, subset, index_mapping):
                self.subset = subset
                self.mapping = index_mapping

            def __getitem__(self, index):
                x, y = self.subset[index]
                return x, self.mapping[y]

            def __len__(self):
                return len(self.subset)

        return RemappedSubset(subset, index_mapping)

    def _loader_to_numpy(self, loader):
        X, y = [], []
        for inputs, labels in loader:
            X.append(inputs.numpy())
            y.append(labels.numpy())
        return np.concatenate(X), np.concatenate(y)

    def _limit_examples_per_class(self, dataset):

        if isinstance(dataset, Subset):
            samples = [dataset.dataset.samples[i] for i in dataset.indices]
            indices = dataset.indices
        else:
            samples = dataset.samples
            indices = list(range(len(samples)))

        class_indices = collections.defaultdict(list)
        for idx, (_, target) in zip(indices, samples):
            class_indices[target].append(idx)
 
        random.seed(self.random_state)
        limited_indices = []
        for target_class, class_idxs in class_indices.items():
            if len(class_idxs) > self.examples_per_class:
                limited_indices.extend(random.sample(class_idxs, self.examples_per_class))
            else:
                limited_indices.extend(class_idxs)
    
        return limited_indices

    def _split_dataset(self, dataset):
        indices_to_split = range(len(dataset))
        if isinstance(dataset, Subset):
            targets = [dataset.dataset.targets[i] for i in dataset.indices]
        else:
            targets = dataset.targets
    
        return train_test_split(
            indices_to_split, 
            test_size=self.test_size,
            random_state=self.random_state, 
            stratify=targets
        )

    def count_images_per_class(self, dataset):
        class_counts = collections.defaultdict(int)
        if isinstance(dataset, Subset):
            samples = [dataset.dataset.samples[i] for i in dataset.indices]
        else:
            samples = dataset.samples
        for _, label in samples:
            class_counts[label] += 1
        return dict(class_counts)

def load_dataset(dataset, output, limit, allowed_classes, image_size, test_size):
    if dataset == "2hojas":
        data = Two_Dataset(
            root='../dataset/hojas2clases',
            image_size=image_size,
            examples_per_class=limit,
            batch_size=32,
            test_size=test_size,
            allowed_classes=allowed_classes,
            output=output,
            dset=dataset
        )

        if output == 'dl':
            return data.get_loaders()
        elif output == 'np':
            X_train, y_train, X_val, y_val, index_mapping = data.get_loaders()

            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.long)

            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)

            return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(val_dataset, batch_size=32, shuffle=False)
    else:
        raise ValueError("Invalid dataset. Accepted values are '2hojas', 'Mariposas', 'Agricultura', 'Arroz' or 'Satelite'")
