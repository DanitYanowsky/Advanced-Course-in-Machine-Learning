from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset

batch_size = 256
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

transform_mnist = transforms.Compose([
    transforms.Resize(32),
    transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
    transforms.ToTensor()])
transform_original = transforms.Compose([transforms.ToTensor()])
def load_data_cifar10(augment):
    if augment:
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_original, download=True)
    else:
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform_original, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=test_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def load_data_mnist():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform_mnist, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform_mnist, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
