import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64):
    """
    Prepara y devuelve los DataLoaders para MNIST.
    """
    root_folder = './data'

    # Transformaciones: Tensor + Normalización (calculada para MNIST)
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
    # Pequeña prueba para ver si funciona
    train_loader, _ = get_data_loaders()
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    print(f"Forma de las imágenes: {images.shape}")
    print(f"Forma de las etiquetas: {labels.shape}")