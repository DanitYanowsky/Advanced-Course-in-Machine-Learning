from VICReg import VICReg
from torchvision import datasets, transforms, models
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from utils import train_transform, load_data_cifar10, test_transform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from LinearProbing import LinearProbing
import json
import torch.nn.functional as F
import numpy as np
import random
import faiss
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

TRAIN = True
PLOTS = True
Q1=True
Q2 = True
Q3 = True
Q4 = True
Q5= True
Q7 = True
gama = 25
mu = 25
v = 1
margin =1
epsilon = 0.0001
d=512
encoder_dimension=128
batch_size = 256
learning_rate = 0.0003
betas =(0.9,0.999)
weight_decay = 0.000001
num_epchos = 35
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
k_pca = 10

def invariance_loss(z, z_tag):
    return F.mse_loss(z, z_tag)
def variance_loss(z):
    sigma = torch.sqrt(z.var(dim=0)+epsilon)
    hinge_loss = torch.mean(torch.relu(1-sigma))
    return hinge_loss
def covariance_loss(z):
    D = z.shape[1]
    cov_matrix = torch.cov(z.T)
    loss = cov_matrix.fill_diagonal_(0).pow(2).sum() / D
    return loss

def log_epoch_losses(epoch, avg_invariance_loss, avg_variance_loss, avg_covariance_loss, avg_total_loss):
    print(f"Epoch [{epoch}] - Invariance Loss: {avg_invariance_loss:.5f}, "
          f"Variance Loss: {avg_variance_loss:.5f}, Covariance Loss: {avg_covariance_loss:.5f}, Total Loss: {avg_total_loss:.5f}")

def train(model, optimizer, train_loader, path='model_weights.pth'):
    loss_history = []

    for epoch in range(num_epchos):
        running_loss = 0.0
        running_invariance_loss = 0.0
        running_variance_loss = 0.0
        running_covariance_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        for i, (inputs, labels) in enumerate(progress_bar):
            x = torch.stack([train_transform(img) for img in inputs]).to(device)
            x_tag= torch.stack([train_transform(img) for img in inputs]).to(device)
            optimizer.zero_grad()

            z = model(x)
            z_tag = model(x_tag)
            invariance_loss_value = gama * invariance_loss(z,z_tag)
            variance_loss_value=  mu * (variance_loss(z) + variance_loss(z_tag))
            covariance_loss_value = v * (covariance_loss(z) + covariance_loss(z_tag))

            total_loss = (invariance_loss_value+variance_loss_value+covariance_loss_value).to(device)
            total_loss.backward()
            optimizer.step()

            running_invariance_loss += invariance_loss_value.item()
            running_variance_loss += variance_loss_value.item()
            running_covariance_loss += covariance_loss_value.item()
            running_loss += total_loss.item()

        avg_invariance_loss = running_invariance_loss / len(train_loader)
        avg_variance_loss = running_variance_loss / len(train_loader)
        avg_covariance_loss = running_covariance_loss / len(train_loader)
        avg_total_loss = running_loss / len(train_loader)

        # Log average losses per epoch
        log_epoch_losses(epoch, avg_invariance_loss, avg_variance_loss, avg_covariance_loss, avg_total_loss)

        loss_history.append({
            'epoch': epoch,
            'invariance_loss': avg_invariance_loss,
            'variance_loss': avg_variance_loss,
            'covariance_loss': avg_covariance_loss,
            'total_loss': avg_total_loss,
        })
    with open(f'{path}.json', 'w') as json_file:
        json.dump(loss_history, json_file)
    save_model(model, epoch, path=path)
    return loss_history
def load_model(model, path='model_weights.pth'): ##chatgpt
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path} at epoch {epoch}")
    return model, epoch
def save_model(model, epoch, path='model_weights.pth'):##chatgpt
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }, path)
    print(f"Model saved at epoch {epoch} to {path}")

def plot_loss(loss_history, title='Losses Over epochs'):##chatgpt
    plt.figure(figsize=(12, 8))

    epochs = [entry['epoch'] for entry in loss_history]
    if 'invariance_loss' in loss_history[0]:
        invariance_losses = [entry['invariance_loss'] for entry in loss_history]
        plt.plot(epochs, invariance_losses, label='Invariance Loss')
    if 'variance_loss' in loss_history[0]:
        variance_losses = [entry['variance_loss'] for entry in loss_history]
        plt.plot(epochs, variance_losses, label='Variance losses')
    if 'covariance_loss' in loss_history[0]:
        covariance_losses = [entry['covariance_loss'] for entry in loss_history]
        plt.plot(epochs, covariance_losses, label='Covariance Loss')
    if 'total_loss' in loss_history[0]:
        plot_value = [entry['total_loss'] for entry in loss_history]
    if 'Accuracy' in loss_history[0]:
        plot_value = [entry['Accuracy'] for entry in loss_history]
    plt.plot(epochs, plot_value, label='Total Loss')
    plt.xlabel('Epoch')
    if 'total_loss' in loss_history[0]:
        plt.ylabel('Loss')
    if 'Accuracy' in loss_history[0]:
        plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def image_representation(data_loader, model):
    model.eval()

    representations = []
    labels_return = []
    progress_bar = tqdm(data_loader, desc=f"", leave=True)
    representations_images =[]
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(progress_bar):
            x = inputs.to(device)
            z = model.test(x).cpu()
            representations.append(z)
            labels_return.append(labels)
            for image in inputs:
                representations_images.append(image)
        representations = torch.cat(representations, dim=0)
        labels_return = torch.cat(labels_return, dim=0)

    return representations, labels_return, representations_images

def pca(z):
    pca_model = PCA(n_components=2)
    return  pca_model.fit_transform(z)
def tsne(z):
    tsne_model = TSNE(n_components=2)
    tsne_results  = tsne_model.fit_transform(z)
    return tsne_results


def plot(representations, title, labels): ##chatgpt
    if len(representations) != len(labels):
        raise ValueError("The length of representations and labels must be the same.")
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(labels)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors):
        idx = (labels == label)
        plt.scatter(representations[idx, 0], representations[idx, 1], s=10, alpha=0.5, color=color, label=str(label))

    plt.title(f"{title} visualization of representations")
    plt.xlabel(f"{title} Component 1")
    plt.ylabel(f"{title} Component 2")
    plt.legend()
    plt.show()
def linear_probing(model, train_representations, labels, test_representations, test_labels):
    for param in model.parameters():
        param.requires_grad = False
    linear_probing_model = LinearProbing(encoder_dimension)
    optimizer = torch.optim.Adam(linear_probing_model.parameters(), lr=0.001)
    loss_history = []
    for epoch in range(num_epchos):
        running_loss = 0.0
        for i in range(0,len(train_representations), batch_size):
            optimizer.zero_grad()
            z = train_representations[i:i+batch_size]
            outputs = linear_probing_model(z)
            loss = nn.CrossEntropyLoss()(outputs, labels[i:i+batch_size])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_total_loss = running_loss / len(train_representations)
        print(f"Epoch [{epoch}] - Loss: {avg_total_loss}")

        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(test_representations), batch_size):
                z = test_representations[i:i + batch_size]
                outputs = linear_probing_model(z)
                _, predicted = torch.max(outputs.data, 1)
                total += labels[i:i + batch_size].size(0)
                correct += (predicted == test_labels[i:i + batch_size]).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch}] - Accuracy: {accuracy}%")

        loss_history.append({
            'epoch': epoch,
            'Accuracy': accuracy,
        })
    return loss_history

def train_ablation2(model, optimizer, train_loader, representation_images, all_representations, path, device, index):
    for epoch in range(1):
        running_loss = 0.0
        running_invariance_loss = 0.0
        running_variance_loss = 0.0
        running_covariance_loss = 0.0
        random_neighbors = []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)
        loss_history = []
        X = np.array(all_representations)
        neighbors =  index.search(X.astype(np.float32), k=3)[1]
        neighbors=NearestNeighbors(n_neighbors=3).fit(train_image_representations).kneighbors(train_image_representations)[1]
        for neighbors_indices in neighbors:
            random_neighbor_index = random.choice(neighbors_indices)
            random_neighbors.append(random_neighbor_index)

        for i, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(device)
            current_batch_size = inputs.size(0)
            optimizer.zero_grad()
            z = model(inputs.to('cuda'))
            z_tag = model(torch.tensor(np.array(representation_images)[random_neighbors[i:i+current_batch_size]]).to('cuda'))
            invariance_loss_value = gama * invariance_loss(z,z_tag)
            variance_loss_value=  mu * (variance_loss(z) + variance_loss(z_tag))
            covariance_loss_value = v * (covariance_loss(z) + covariance_loss(z_tag))

            total_loss = (invariance_loss_value+variance_loss_value+covariance_loss_value).to(device)
            total_loss.backward()
            optimizer.step()

            running_invariance_loss += invariance_loss_value.item()
            running_variance_loss += variance_loss_value.item()
            running_covariance_loss += covariance_loss_value.item()
            running_loss += total_loss.item()

        avg_invariance_loss = running_invariance_loss / len(train_loader)
        avg_variance_loss = running_variance_loss / len(train_loader)
        avg_covariance_loss = running_covariance_loss / len(train_loader)
        avg_total_loss = running_loss / len(train_loader)

        # Log average losses per epoch
        log_epoch_losses(epoch, avg_invariance_loss, avg_variance_loss, avg_covariance_loss, avg_total_loss)

        loss_history.append({
            'epoch': epoch,
            'invariance_loss': avg_invariance_loss,
            'variance_loss': avg_variance_loss,
            'covariance_loss': avg_covariance_loss,
            'total_loss': avg_total_loss,
        })
    save_model(model, epoch, path=path)
    return loss_history

def unnormalize(tensor, mean, std): ##chatgpt
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor
def plot_neighbors(class_images, images, neighbors_indices, title): ##chatgpt
    plt.figure(figsize=(15, 15))
    num_classes = len(class_images)
    num_neighbors = neighbors_indices.shape[1]

    for i in range(num_classes):
        for j in range(num_neighbors + 1):  # +1 for the original image
            ax = plt.subplot(num_classes, num_neighbors + 1, i * (num_neighbors + 1) + j + 1)
            if j == 0:
                img = class_images[i][0][0].numpy().transpose((1, 2, 0))
                ax.set_title("Original")
            else:
                img = images[neighbors_indices[i, j - 1]].numpy().transpose((1, 2, 0))
                ax.set_title(f"Neighbor {j}")
            img = img * np.array((0.247, 0.243, 0.261)) + np.array((0.4914, 0.4822, 0.4465))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.axis('off')
    plt.suptitle(title)
    plt.show()

def calculate_distances(all_representations):
    X = np.array(all_representations)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X.astype(np.float32))
    return index


if __name__ == "__main__":
    VICReg_model = VICReg(encoder_dimension)
    train_loader, test_loader = load_data_cifar10(augment=True)
    optimizer = torch.optim.Adam(VICReg_model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    if Q1:
        loss_history = train(VICReg_model, optimizer, train_loader, path='model_weights.pth')
        plot_loss(loss_history)
    VICReg_model.eval()
    load_model(VICReg_model)
    train_loader, test_loader = load_data_cifar10(augment=False)
    train_loader.dataset.transform = test_transform  # Ensure no augmentation during evaluation

    train_loader.shuffle = False

    train_image_representations, train_labels, images_original = image_representation(train_loader, VICReg_model)
    test_image_representations, test_labels, _ = image_representation(test_loader, VICReg_model)
    index = calculate_distances(train_image_representations)
    ####Q7-The ablation2#####

    class_images = {}
    num_classes = 10

    for i in range(len(train_labels)):
        if train_labels[i].item() not in class_images:
            class_images[train_labels[i].item()] = (
            images_original[i].unsqueeze(dim=0), torch.tensor(train_labels[i]).unsqueeze(dim=0))
        if len(class_images) == num_classes:
            break
    class_images = list(class_images.values())
    representations_ablation2 = image_representation(class_images, VICReg_model)[0]
    neighbors = \
    NearestNeighbors(n_neighbors=len(train_image_representations)).fit(train_image_representations).kneighbors(
        train_image_representations)[1]
    closest_neighbors = neighbors[:, 1:6]
    distant_neighbors = neighbors[:, -5:]
    plot_neighbors(class_images, images_original, closest_neighbors, "VICREG - 5 closest neighbors")
    plot_neighbors(class_images, images_original, distant_neighbors, "VICREG - 5 distant neighbors")

    if Q2:
        pca_representations = pca(test_image_representations)
        tsne_representations = tsne(test_image_representations)
        plot(pca_representations, "pca", test_labels)
        plot(tsne_representations, "tsne", test_labels)
    if Q3:
        loss_history_classifier = linear_probing(VICReg_model, train_image_representations, train_labels, test_image_representations, test_labels)
        plot_loss(loss_history_classifier, "linear probing")

    if Q4:
        mu = 0
        train_loader_q4, test_loader = load_data_cifar10(augment=True)
        model_ablation1 = VICReg(encoder_dimension)
        optimizer = torch.optim.Adam(model_ablation1.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        loss_history_ablation1 = train(model_ablation1, optimizer, train_loader_q4, path='model_weights_Ablation1.pth')
        load_model(model_ablation1, path='model_weights_Ablation1.pth')
        plot_loss(loss_history_classifier, "linear probing")
        train_image_representations_ablation1, train_labels_ablation1, _ = image_representation(train_loader, model_ablation1)
        test_image_representations_ablation1, test_labls_ablation1, _ = image_representation(test_loader, model_ablation1)
        loss_history_classifier = linear_probing(model_ablation1, train_image_representations_ablation1, train_labels_ablation1
                                                 , test_image_representations_ablation1, test_labls_ablation1)
        plot_loss(loss_history_classifier, "linear probing - ablation 1")
        pca_representations_ablation1 = pca(test_image_representations_ablation1)
        tsne_representations_ablation1 = tsne(test_image_representations_ablation1)
        plot(pca_representations_ablation1, "pca", test_labels)
        plot(tsne_representations_ablation1, "tsne", test_labels)
    if Q5:
        mu=25
        train_loader_q5, test_loader_q5 = load_data_cifar10(augment=False)
        model_abaltion2 = VICReg(encoder_dimension)
        optimizer = torch.optim.Adam(model_abaltion2.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
        if TRAIN:
            train_ablation2(model_abaltion2, optimizer, train_loader_q5 ,images_original, train_image_representations,path='model_weights_Ablation2.pth', device=device, index=index)
        load_model(model_abaltion2, path='model_weights_Ablation2.pth')
        train_image_representations_ablation2, train_labels_ablation2, images = image_representation(train_loader_q5, model_abaltion2)
        test_image_representations_ablation2, test_labels_ablation2, _ = image_representation(test_loader_q5, model_abaltion2)
        loss_history_classifier = linear_probing(model_abaltion2, train_image_representations_ablation2, train_labels_ablation2,
                                                  test_image_representations_ablation2, test_labels_ablation2)
        plot_loss(loss_history_classifier, "linear probing - ablation 2")
        pca_representations_ablation2 = pca(test_image_representations_ablation2)
        tsne_representations_ablation2 = tsne(test_image_representations_ablation2)
        plot(pca_representations_ablation2, "pca - ablation 2", test_labels)
        plot(tsne_representations_ablation2, "tsne - ablation 2", test_labels)
        ####Q7-The ablation2#####
        representations_ablation2 = image_representation(class_images, model_abaltion2)[0]
        neighbors = NearestNeighbors(n_neighbors=len(train_image_representations_ablation2)).fit(train_image_representations_ablation2).kneighbors(train_image_representations_ablation2)[1]
        closest_neighbors = neighbors[:, 1:6]
        distant_neighbors = neighbors[:, -5:]
        plot_neighbors(class_images,images, closest_neighbors, "Ablation2 - 5 closest neighbors")
        plot_neighbors(class_images,images, distant_neighbors, "Ablation2 - 5 distant neighbors")


