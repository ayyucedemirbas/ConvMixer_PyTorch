import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


def get_transforms(img_height, img_width):
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def create_datasets(train_path, test_path, train_transform, test_transform):

    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)

    test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    return train_dataset, val_dataset, test_dataset


def create_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
