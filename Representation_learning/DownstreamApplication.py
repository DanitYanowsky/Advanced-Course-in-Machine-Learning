from Train import load_model, encoder_dimension, image_representation, tsne
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import faiss
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
from utils import load_data_mnist, load_data_cifar10
from VICReg import VICReg
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
def cluster_representations(representations, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(representations)
    return kmeans.labels_, kmeans.cluster_centers_


def clustering(representations, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42) ##chatGPT
    cluster_labels = kmeans.fit_predict(representations)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

def visualize_clusters(tsne_result, labels, cluster_labels, cluster_centers, title): ##chatgpt
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='tab10', s=2)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='X', s=100)
    plt.title(f'{title} - T-SNE (Cluster Labels)')
    plt.subplot(1, 2, 2)
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', s=2)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='X', s=100)
    plt.title(f'{title} - T-SNE (True Labels)')
    plt.show()

def unnormalize(tensor, mean, std):
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    tensor = tensor * std + mean
    return tensor

def plot_iamges(images, title):
    plt.figure(figsize=(15, 15))
    num_images = len(images)

    for i in range(num_images):
        ax = plt.subplot(num_images, 6, i * 6 + 1)
        if images[i][0].shape[0] == 3:
            img = images[i][0].numpy().transpose((1, 2, 0))
            img = unnormalize(img, mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
            img = np.clip(img, 0, 1)
            ax.imshow(img)
        else:
            ax.imshow(images[i][0], cmap='gray')
        ax.set_title(f"Image {i}")
        ax.axis('off')
    plt.suptitle(title)
    plt.show()

VICReg_trained, VICReg_ablation2 = VICReg(encoder_dimension), VICReg(encoder_dimension)
_, test_loader_mnist = load_data_mnist()
load_model(VICReg_trained)
load_model(VICReg_ablation2, path='model_weights_Ablation2.pth')
def plot_ruc(auc, fpr, tpr): ##chatGPT
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC , AUC:{auc}')
    plt.legend(loc="lower right")
    plt.show()

def Q2(model, title, augment=False):
    train_loader, test_loader_cifar10 = load_data_cifar10(augment=False)
    train_image_representations, labels, images = image_representation(train_loader, model)
    test_image_representations_cifar10, _, images_cifar10 = image_representation(test_loader_cifar10, model)
    test_image_representations_mnist, _, images_mnist = image_representation(test_loader_mnist, model)
    combined_representations = np.concatenate((test_image_representations_cifar10, test_image_representations_mnist),
                                              axis=0)
    combined_labels = np.concatenate((torch.zeros(len(test_image_representations_cifar10)),
                                      torch.ones(len(test_image_representations_mnist))), axis=0)
    density_score = NearestNeighbors(n_neighbors=2).fit(train_image_representations).kneighbors(combined_representations)[0].mean(axis=1)
    fpr, tpr, _ = roc_curve(y_true=combined_labels, y_score=density_score)
    plot_ruc(roc_auc_score(combined_labels, density_score), fpr, tpr)

    anomaly = np.argsort(density_score)[-30:]
    images =[]
    for i in anomaly:
        if i >10000:
            images.append(datasets.MNIST(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)[i-10000][0])
        else:
            images.append(images_cifar10[i])
    plot_iamges(images, 'Anomaly Detection')
    ###Clustering###
    tsne_rep = tsne(train_image_representations)
    cluster_label, cluster_center = clustering(train_image_representations)
    labels, centers_ablation = cluster_representations(train_image_representations)
    visualize_clusters(tsne_rep, labels, cluster_label, tsne_rep[-10:], "VICReg")
    from sklearn.metrics import silhouette_score

    silhouette_score = silhouette_score(train_image_representations, labels)
    print(f"Silhouette Score for {title}: {silhouette_score:.4f}")


Q2(VICReg_trained, title='VICReg_trained', augment=False,)
Q2(VICReg_ablation2, title='VICReg_ablation2', augment=False,)





