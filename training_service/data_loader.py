import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    Prepares and returns the DataLoaders for MNIST.
    """
    # Note: This path is relative to where the script is run (e.g., 'src/data')
    # If running 'api.py' from 'src/', it will create 'src/data/'
    # If running 'main.py' from 'app/', it will create 'app/data/'
    # For consistency, you could use an absolute path based on PROJECT_ROOT
    root_folder = './data'

    # Transformations: Tensor + Normalization (pre-calculated for MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Datasets
    train_dataset = torchvision.datasets.MNIST(
        root=root_folder,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root=root_folder,
        train=False,
        download=True,
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    # A small test to see if it works
    train_loader, _ = get_data_loaders()
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")