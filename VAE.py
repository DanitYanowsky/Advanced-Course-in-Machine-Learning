import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# take a stratified subset of the training data, keeping only 5000 samples, with 500 samples per class
train_targets = train_dataset.targets
train_idx, _ = train_test_split(range(len(train_targets)), train_size=20000, stratify=train_targets)
train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def load_checkpoint(epoch, model, optimizer, type_of_model): ##used ChatGPT
    checkpoint_dir = f'/cs/labs/daphna/danit.yanowsky/CL/{type_of_model}/epoch_{epoch}'
    if os.path.exists(checkpoint_dir):
        checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint_.pth")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from epoch {epoch}")

def save_checkpoint(epoch, model, optimizer, type_of_model): ##used ChatGPT
    checkpoint_dir = f'/cs/labs/daphna/danit.yanowsky/CL/{type_of_model}/epoch_{epoch}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'/cs/labs/daphna/danit.yanowsky/CL/{type_of_model}/epoch_{epoch}/model_checkpoint_.pth')
        print("Checkpoint saved successfully.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
def select_images_per_class(validation_data, num): ##used ChatGPT
    images_per_class = {}
    result={}
    index=0
    for images, labels in validation_data:
        for image, label in zip(images, labels):
            label_item = label.item()
            if label_item not in images_per_class:
                images_per_class[label_item] = 0
                result[label_item] = []
            if images_per_class[label_item]<num:
                images_per_class[label_item]+=1
                result[label_item].append(image)
            index+=1
    # Convert lists of images into tensors
    for label_item, images in result.items():
        result[label_item] = torch.stack(images)

    return result



def display_images(images, dataset_type, model, epoch): ##used ChatGPT
    num_classes = len(images)
    num_images_per_class = 1
    if num_classes == 1:
        fig, ax = plt.subplots(1, num_images_per_class, figsize=(5, 5))
        ax = [ax]  # Convert to list for uniformity
    else:
        fig, ax = plt.subplots(num_classes, num_images_per_class, figsize=(10, 10))

    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    for i in range(num_classes):
        # Reshape the image if it's flattened
        image = images[i].view(28, 28).detach().numpy()  # Assuming MNIST images are not flattened
        image = image.astype('float32')  # Convert to float32
        # Normalize pixel values to range [0, 1]
        # Pass the image through the model and reshape
        ax[i].imshow(image, cmap='gray')  # Specify cmap='gray' for grayscale images
        ax[i].axis('off')
        ax[i].set_title(f'Epoch: {epoch}, {dataset_type} Image, label {i}')
    plt.show()

class VAELatentOptimization(nn.Module):
    def __init__(self, latent_dim=200, num_samples=10):
        super(VAELatentOptimization, self).__init__()

        self.latent_dim = latent_dim

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)
        self.mu = torch.randn(num_samples * latent_dim, requires_grad=True)
        self.logvar = torch.randn(num_samples * latent_dim, requires_grad=True)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # (batch_size, 128, 2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 1, 28, 28)
        )

    def reparameterize(self, start, end, image_per_batch):
        ################## YOUR CODE HERE ######################
        std = torch.exp(0.5 * self.logvar[start:end])
        identity_matrix = torch.ones(end - start)
        epsilon = torch.normal(0, identity_matrix)
        return self.mu[start:end] + std * epsilon.unsqueeze(0)
        ########################################################

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), z.size(1), 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, start, end, image_per_batch):
        z = self.reparameterize(start, end, image_per_batch)
        z = z.view(image_per_batch, self.latent_dim)
        recon_x = self.decode(z)
        return recon_x



def latent_optimization():
    model = VAELatentOptimization(num_samples=len(train_loader.dataset.indices))
    criterion = lambda outputs, inputs, mu, sigma: torch.sum(torch.norm(inputs-outputs, p=2, dim=(1,2,3)) + torch.mean(
        (sigma ** 2 + (mu) ** 2) - torch.log(sigma) - 1, dim =1)) ##used chatgpt
    # # Training loop
    val_losses = []
    val_accuracies = []
    num_epochs = 30
    mu_log_parameters = [{'params': model.mu}, {'params': model.logvar}]
    parameters_to_optimize = [
        *mu_log_parameters,
        {"params": model.parameters(), "lr": 0.001}
    ]
    # Define the optimizer
    optimizer = optim.Adam(parameters_to_optimize, lr=0.01)
    for epoch in range(num_epochs):
        num_batch = 0
        running_loss = 0.0

        for image_batch, labels in tqdm(train_loader):
            model.train()  # moves the model to training mode
            optimizer.zero_grad()
            start = num_batch * image_batch.size()[0] * model.latent_dim
            end = (num_batch + 1) * image_batch.size()[0] * model.latent_dim
            output = model(start, end, image_batch.size()[0])
            sigma = torch.sqrt(torch.exp(model.logvar[start:end]))
            mu =torch.stack(torch.split(model.mu[start:end], split_size_or_sections=200)) ##used ChatGPT
            sigma =torch.stack(torch.split(sigma, split_size_or_sections=200)) ##used ChatGPT
            losses = criterion(image_batch, output, mu, sigma)
            losses.backward()
            optimizer.step()
            epochs_list = [0, 4, 9, 19, 29]
            num_batch += 1
        if epoch in epochs_list:
            save_checkpoint(epoch, model, optimizer, "latent")

        val_losses.append(running_loss)


def plot_latent_optimization(epochs, list_model, selected_train_images_dict): ##used ChatGPT

    display_images_in_row_latent(selected_train_images_dict, 'Reconstruction Train', list_model,epochs)
    latent_images_dict = {}
    for j in range(len(list_model)):
        for _ in range(10):
            identity_matrix = torch.ones(list_model[j].latent_dim)
            z = torch.normal(0, identity_matrix)
            output = list_model[j].decode(z.view(1, z.size(0)))
            latent_images_dict.update({_: output})
        plot_prior_images(j, list_model[j])


def display_images_in_row_amortized(images_dict, dataset_type, models, epochs): ##used ChatGPT
    num_images = len(images_dict)
    fig, axes = plt.subplots(num_images, len(models)+1, figsize=(15, 3 * num_images))

    for i, (index, image) in enumerate(images_dict.items()):
        # Show original image
        axes[i, 0].imshow(image[0].permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
        axes[i, 0].set_title(f'{dataset_type} - Digit {index}')
        axes[i, 0].axis('off')
        for j in range(len(models)):
            model_image = models[j](image)[0][0]
            axes[i, j+1].imshow(model_image.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[i, j+1].set_title(f'Reconstruction epoch {epochs[j]+1}')
            axes[i, j+1].axis('off')

    plt.tight_layout()
    plt.show()

def display_images_in_row_latent(images_dict, dataset_type, models, epochs):  ##used ChatGPT
    num_images = len(images_dict)
    fig, axes = plt.subplots(num_images, len(models) + 1, figsize=(15, 3 * num_images))

    for i, (index, image) in enumerate(images_dict.items()):
        # Show original image
        axes[i, 0].imshow(image[0].permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
        axes[i, 0].set_title(f'{dataset_type} - Digit {index}')
        axes[i, 0].axis('off')
        for j in range(len(models)):
            model_image = models[j](i * 200, (i + 1) * 200, 1)[0]
            axes[i, j + 1].imshow(model_image.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[i, j + 1].set_title(f'Reconstruction epoch {epochs[j] + 1}')
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()
def display_images_log_prob(images, values, title): ##used ChatGPT
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=16)
    for j in range(5):
        axes[j].imshow(images[j].permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
        axes[j].axis('off')
        axes[j].set_title(f'Log Probability {torch.round(values[j])}')
    plt.savefig(f'log_prob_{title}.png')


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=200):
        super(ConvVAE, self).__init__()

        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=2)  # (batch_size, 512, 1, 1)
        )

        # Latent space
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=2),  # (batch_size, 128, 2, 2)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 128, 4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (batch_size, 64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            # (batch_size, 1, 28, 28)
        )

    def reparameterize(self, mu, logvar):
        ################## YOUR CODE HERE ######################
        ##transform r to std:
        std = torch.exp(0.5 * logvar)
        identity_matrix = torch.ones(self.latent_dim)
        epsilon = torch.normal(0, identity_matrix)
        return mu + std * epsilon
        ########################################################

    def encode(self, x):
        x = self.encoder(x)
        # add average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def decode(self, z):
        z = self.fc_decode(z)
        z = z.view(z.size(0), 128, 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_amortized(display_images, selected_train_images_dict, selected_test_images_dict):

    model = ConvVAE()
    criterion = lambda outputs, inputs, mu, sigma: torch.sum(nn.MSELoss()(outputs, inputs) + torch.mean(
        (sigma ** 2 + (mu) ** 2) - torch.log(sigma) - 1, dim=1))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop
    val_losses = []
    train_losses = []
    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()  # moves the model to training mode
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            ################### Complete the code below ###################
            optimizer.zero_grad()
            outputs, mu, logvar = model(images)
            sigma = torch.sqrt(torch.exp(logvar))
            loss = criterion(outputs, images, mu, sigma)
            loss.backward()
            optimizer.step()
            ###############################################################
            running_loss += loss.item()

        # Validation
        model.eval()  # moves the model to evaluation mode
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():  # Temporarily set all the requires_grad flags to false
            for images, labels in tqdm(test_loader):
                outputs, mu, logvar = model(images)
                sigma = torch.sqrt(torch.exp(logvar))
                loss = criterion(outputs, images, mu, sigma)
                val_loss += loss.item()
                total += labels.size(0)
        epochs_list = [0, 4, 9, 19, 29]
        if epoch in epochs_list:
            save_checkpoint(epoch, model, optimizer,"amortized")
        epoch_loss = running_loss / len(train_loader)
        val_loss /= len(test_loader)
        val_losses.append(val_loss)
        train_losses.append(epoch_loss)
    ################### Complete the code below ###################
    # Plot validation loss
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red', marker='o')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot test loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', color='blue', marker='o')
    plt.title('Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
    ###############################################################

def plot_prior_images(epoch, model):
    epochs_list = [0,5,10,20,30]
    latent_images_dict = {}
    for _ in range(10):
         identity_matrix = torch.ones(model.latent_dim)
         z = torch.normal(0, identity_matrix)
         output = model.decode(z.view(1, z.size(0)))
         latent_images_dict.update({_: output.squeeze(0)})
    fig, axes = plt.subplots(10, 1, figsize=(15, 3 * 10))

    for i, (index, image) in enumerate(latent_images_dict.items()):
            model_image =image
            axes[i].imshow(model_image.permute(1, 2, 0).cpu().detach().numpy(), cmap='gray')
            axes[i].set_title(f'Epoch {epochs_list[epoch]} - Latent Vectors Sampled from Prior Distribution')
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def compute_log_probability(model, image, M):
    p_sigma = 0.4
    mu, logvar = model.encode(image)
    image_flatten = torch.flatten(image, start_dim=1)
    z = [model.reparameterize(mu, logvar) for i in range(M)]
    recon_images = [model.decode(z[i]) for i in range(M)]
    for i in range(M):
        recon_image_flatten =  torch.flatten(recon_images[i], start_dim=1)
        sigma =torch.ones(image.size()[0],28*28) * np.log(p_sigma * p_sigma)
        p_x_z = log_normal_pdf(image_flatten, recon_image_flatten, sigma)
        q_z = log_normal_pdf(z[i], mu, logvar)
        p_z = log_normal_pdf(z[i], torch.zeros(z[i].size()), torch.zeros(z[i].size()))
    p_x_z, q_z, p_z=  p_x_z.view(p_x_z.size()[0],1), q_z.view(q_z.size()[0],1), p_z.view(p_z.size()[0],1) ## used chatgpt
    return torch.logsumexp(p_x_z + q_z - p_z, dim=1) - torch.log(torch.tensor(M, dtype=torch.float))


def log_normal_pdf(x, mu, logvar): ##used ChatGPT
    # Logarithm of the normalizing constant
    log_norm_const = -0.5 * (x.size(1) * (math.log(2 * math.pi)) + torch.sum(logvar, dim=1))
    # Exponent term
    exponent = -0.5 * torch.sum(((x - mu)**2) / torch.exp(logvar), dim=1)
    # Log normal PDF
    log_pdf = log_norm_const + exponent
    return log_pdf

def combine_dicts(dict1, dict2): ## used ChatGPT
    combined_dict = dict1.copy()  # Make a copy of the first dictionary

    for key, value in dict2.items():
        if key in combined_dict:
            combined_dict[key] += value  # Add values if key already exists
        else:
            combined_dict[key] = value  # Otherwise, add the key-value pair

    return combined_dict
def show_image_logprob(images_dict, model, title):
    log_probability = {}
    average_prob_per_digit = {}
    for i, images in images_dict.items():
        log_probability[images]= compute_log_probability(model, images,  1000)
        average_prob_per_digit[i] = torch.mean(log_probability[images])
        display_images_log_prob(images, log_probability[images], title + f" - digit {i}")
    return log_probability, average_prob_per_digit


######For all the Questions######

epochs = [0,4,9,19,29]
selected_test_images_dict = select_images_per_class(test_loader, 1)
selected_train_images_dict = select_images_per_class(train_loader, 1)

######Q1+2######
#train the model
vae_amortized(display_images, selected_train_images_dict, selected_test_images_dict)
list_of_models=[]
for i in epochs:
    model_amortized = ConvVAE()
    optimizer = optim.Adam(model_amortized.parameters(), lr=0.001)
    load_checkpoint(i, model_amortized, optimizer, "amortized")
    list_of_models.append(model_amortized)
    plot_prior_images(i, model_amortized)
display_images_in_row_amortized(selected_train_images_dict, 'Train', list_of_models, epochs)
display_images_in_row_amortized(selected_test_images_dict, 'Validation', list_of_models, epochs)
#####Q3######

latent_optimization()
list_of_models=[]


for i in epochs:
    model_latent = VAELatentOptimization(num_samples=len(train_loader.dataset.indices))
    mu_log_parameters = [{'params': model_latent.mu}, {'params': model_latent.logvar}]
    parameters_to_optimize = [
        *mu_log_parameters,
        {"params": model_latent.parameters(), "lr": 0.001}
    ]
    # Define the optimizer
    optimizer = optim.Adam(parameters_to_optimize, lr=0.01)

    load_checkpoint(i, model_latent, optimizer, "latent")
    list_of_models.append(model_latent)

plot_latent_optimization(epochs, list_of_models, selected_test_images_dict)

#########################Q4#########################
#####Q4.a######
train_five = select_images_per_class(train_loader, 5)
log_prob_train, average_prob_per_digit_train =show_image_logprob(train_five, model_amortized, "Train images")
test_five = select_images_per_class(test_loader, 5)
log_prob_test, average_prob_per_digit_test = show_image_logprob(test_five, model_amortized,"Test images")
######Q4.b######
for i, a in average_prob_per_digit_train.items():
    print(f"Average log-probability for digit {i} is {(a + average_prob_per_digit_test[i])/2}")

######Q4.c######
average_dict_values = lambda input_dict: sum(input_dict.values()) / len(input_dict) ## used ChatGpt
print(f"Average log-probability for train images: {average_dict_values(average_prob_per_digit_train)}" )
print(f"Average log-probability for test images: {average_dict_values(average_prob_per_digit_test)}")
