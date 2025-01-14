import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from pytorch_lightning import seed_everything
from create_data import create_olympic_rings

###### model #####
seed_everything(0)
class FlowMatching(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_t=0.001
        self.fc1 = nn.Linear(input_dim+1+1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.class_condition = torch.nn.Embedding(5, 3)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, y,t,class_condition):
        y = torch.cat([y,t, class_condition],dim=1)
        y = nn.LeakyReLU()(self.fc1(y))
        y = nn.LeakyReLU()(self.fc2(y))
        y = nn.LeakyReLU()(self.fc3(y))
        y = nn.LeakyReLU()(self.fc4(y))
        y = nn.LeakyReLU()(self.fc5(y))
        y = self.fc6(y)
        return y

def train_flow_model(model, loss_array, t, model_save_path, optimizer_save_path):
    len_data = len(train_data[0])
    data, labels = train_data[0], train_data[1]
    shuffle = np.random.permutation(len_data)
    data, labels = data[shuffle], labels[shuffle]
    for i in range(epochs):
        running_loss = 0.0
        for j in range(0, len_data, batch_size):
            y_1, label = torch.tensor(data[j:j + batch_size], dtype=torch.float32), torch.tensor(labels[j:j + batch_size], dtype=torch.float32)
            current_batch_size = len(y_1)
            y_0 = torch.randn(y_1.shape)
            t = torch.rand(current_batch_size).view(-1, 1)  ##chat GPT
            y = (1 - t) * y_0 + t * y_1
            v_t = V(y, t, label.view(-1,1))
            loss = ((v_t - (y_1 - y_0)) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss+=loss.item()
        avg_val_loss = running_loss / len(train_data[0])
        print(f'Epoch: {i + 1}/{epochs}, Loss: {avg_val_loss}')
        loss_array.append(avg_val_loss)
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)
########### Hyperparameters ###########
t=0
epochs=20
batch_size=128
learning_rate=1e-3
number_of_data_points=250000
validation_size=number_of_data_points//10
#######################################
############# Initialize ##############

V = FlowMatching()
optimizer = torch.optim.Adam(V.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_array=[]
train_data = create_olympic_rings(number_of_data_points,verbose=True)

#######################################
def plot_sampled_points(sampled_points, labels, int_to_label):
    train_data = create_olympic_rings(number_of_data_points, verbose=True)


def Q1():
    plot_sampled_points(train_data[0], train_data[1], train_data[2])
    plot_loss()
def plot_loss(): ##chatGPT
    plt.figure(figsize=(10, 6))
    plt.plot(loss_array, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def Q2():
    global timesteps, all_labels, t, flat_trajectories
    timesteps = 1000
    labels = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).view(-1,1)
    trajectory = torch.randn(5, 2, dtype=torch.float32)
    all_trajectories = []
    for t in np.arange(0, 1+V.delta_t, V.delta_t):
        velocity = V(trajectory, torch.tensor(t, dtype=torch.float32).repeat(5).unsqueeze(dim=1), labels).detach()
        trajectory += velocity * V.delta_t
        all_trajectories.append(trajectory.numpy().copy())
    all_trajectories = np.array(all_trajectories)
    plot_trajectories(all_trajectories, labels, train_data[2])
def plot_trajectories(all_trajectories, all_labels, label_color_map): ##chat GPT
    norm = plt.Normalize(vmin=0, vmax=len(all_trajectories) - 1)
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    for i in range(5):
        trajectory = all_trajectories[:, i, :]
        time_steps = np.arange(trajectory.shape[0])
        label = all_labels[i]
        for j in range(trajectory.shape[0] - 1):
            plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1],
                     color=cmap(norm(j)), alpha=0.5)
        plt.scatter(trajectory[:, 0], trajectory[:, 1],
                    c=time_steps, cmap=cmap, norm=norm, marker='o', alpha=0.5)
        plt.text(trajectory[-1, 0], trajectory[-1, 1], f'{i}',
                 fontsize=12, color=label_color_map.get(label.item()), weight='bold')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory of Points Over Time with NN Velocity')
    plt.grid(True)
    plt.colorbar(label='Timestep')
    plt.show()

def Q3():
    global t
    num_samples = 3000
    array_x = []
    labels_array = []
    with torch.no_grad():
        labels = torch.randint(0, 5, (num_samples,), dtype=torch.float32).view(-1,1)
        samples =torch.randn(num_samples, 2, dtype=torch.float32)
        y=samples
        for t in np.arange(0, 1+0.001, 0.001): ##chatgpt
            v_t = V(y, torch.tensor(t, dtype=torch.float32).repeat(num_samples, 1), labels)
            y = y + v_t * 0.001
        array_x.append(y.detach().numpy())
        labels_array.append(labels)
    array_x = np.concatenate(array_x)
    labels_array = np.concatenate(labels_array)
    plot_sampled_points(array_x, labels_array, train_data[2])

V.eval()
model_save_path = 'model_weights_conditional_flow_matching.pth'
optimizer_save_path = 'optimizer_state_conditional_flow_matching.pth'
#train_flow_model(V, loss_array, t, model_save_path, optimizer_save_path)
V.load_state_dict(torch.load(model_save_path))
optimizer.load_state_dict(torch.load(optimizer_save_path))
Q1()
Q2()
Q3()
