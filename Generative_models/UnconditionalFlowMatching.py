import numpy as np
import torch
from lightning_fabric import seed_everything
from matplotlib import cm
from torch import nn, optim
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize

from sklearn.model_selection import train_test_split
from create_data import create_unconditional_olympic_rings
########### Hyperparameters ###########
seed_everything(0)

t=0
delta_t=0.001
epochs=20
batch_size=128
learning_rate=0.001
number_of_data_points=250000
validation_size=number_of_data_points//10
#######################################
class FlowMatching(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_t = 0.001
        self.fc1 = nn.Linear(input_dim + 1 , hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, output_dim)

    def forward(self, y, t):
        y = torch.cat([y, t], dim=1)
        y = nn.LeakyReLU()(self.fc1(y))
        y = nn.LeakyReLU()(self.fc2(y))
        y = nn.LeakyReLU()(self.fc3(y))
        y = nn.LeakyReLU()(self.fc4(y))
        y = nn.LeakyReLU()(self.fc5(y))

        y = self.fc6(y)
        return y

############# Initialize ##############

V = FlowMatching()
optimizer = torch.optim.Adam(V.parameters(), lr=learning_rate)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
loss_array=[]

################ Data #################
train_data =  torch.tensor(create_unconditional_olympic_rings(number_of_data_points,verbose=False), dtype=torch.float32)

# train_data, validation_data = train_test_split(np.array(data), test_size=validation_size)
# train_data, validation_data = torch.tensor(train_data), torch.tensor(validation_data)

#######################################
def train_flow_model(model, loss_array, t, model_save_path, optimizer_save_path):
    len_data = len(train_data)
    data = train_data
    shuffle = np.random.permutation(len_data)
    data = data[shuffle]
    for i in range(epochs):
        running_loss = 0.0


        for j in range(0, len_data, batch_size):
            y_1 = torch.tensor(data[j:j + batch_size], dtype=torch.float32)
            current_batch_size = len(y_1)
            y_0 = torch.randn(y_1.shape)
            t = torch.rand(current_batch_size).view(-1, 1)  ##chat GPT
            y = (1 - t) * y_0 + t * y_1
            v_t = V(y, t)
            loss = ((v_t - (y_1 - y_0)) ** 2).mean()
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            running_loss+=loss.item()

        avg_val_loss = running_loss / len(train_data)
        loss_array.append(avg_val_loss)
        print(f'Epoch: {i + 1}/{epochs}, Loss: {avg_val_loss}')
    torch.save(model.state_dict(), model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)

def Q1(loss_array):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_array, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def Q2():
    num_samples =1000
    y =torch.randn(num_samples, 2, dtype=torch.float32)
    array_x, array_y=[],[]
    t_array = [0, 0.2, 0.4, 0.6, 0.8, 1]
    with torch.no_grad():
        for t in np.arange(0, 1 + V.delta_t, V.delta_t):
            v_t = V(y, torch.tensor(t, dtype=torch.float32).repeat(num_samples, 1)).detach()
            y = y + v_t * V.delta_t
            if t in t_array:
                array_x.append(y[:,0])
                array_y.append(y[:,1])
    for i, t in enumerate(t_array):
        plt.figure(figsize=(10, 5))
        plt.title(f'Plot for t={t}')
        plt.scatter(array_x[i], array_y[i], c='blue', alpha=0.5)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.show()
model_save_path = 'model_weights_flow_matching.pth'
optimizer_save_path = 'optimizer_state_flow_matching.pth'
train_flow_model(V, loss_array, t, model_save_path, optimizer_save_path)
V.load_state_dict(torch.load(model_save_path))
optimizer.load_state_dict(torch.load(optimizer_save_path))


def Q3():
    global t
    timesteps = 1000
    num_samples = 10
    all_trajectories=[]
    trajectory = torch.randn(num_samples, 2, dtype=torch.float32)
    for t in np.arange(0, 1 + V.delta_t, V.delta_t):
        velocity = V(trajectory, torch.tensor(t, dtype=torch.float32).repeat(num_samples).unsqueeze(dim=1)).detach()
        trajectory += velocity * V.delta_t
        all_trajectories.append(trajectory.numpy().copy())
    all_trajectories = np.array(all_trajectories)
    plot_trajectory(all_trajectories, num_samples)


def plot_trajectory(all_trajectories, num_samples): ##chatGPT
    # Plot trajectories
    norm = plt.Normalize(vmin=0, vmax=len(all_trajectories) - 1)
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis  # Using a single colormap
    for i in range(num_samples):
        trajectory = all_trajectories[:, i, :]
        time_steps = np.arange(trajectory.shape[0])

        # Plot each segment of the trajectory with varying brightness
        for j in range(trajectory.shape[0] - 1):
            plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1],
                     color=cmap(norm(j)), alpha=0.5)

        # Add a scatter plot to show the points with varying brightness
        plt.scatter(trajectory[:, 0], trajectory[:, 1],
                    c=time_steps, cmap=cmap, norm=norm, marker='o', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory of Points Over Time with NN Velocity')
    plt.grid(True)
    plt.colorbar(label='Timestep')
    plt.show()


def Q4():
    global t
    delta_t_array = [0.002, 0.02, 0.05, 0.1, 0.2]
    gaussian = MultivariateNormal(torch.zeros(2), torch.eye(2))

    num_samples=1000
    for delta_t in delta_t_array:
        x_array = []
        y_array = []
        V.delta_t = delta_t
        samples = torch.randn(num_samples, 2, dtype=torch.float32)
        for t in np.arange(0, 1 + V.delta_t, V.delta_t):
            velocity = V(samples, torch.tensor(t, dtype=torch.float32).repeat(num_samples).unsqueeze(dim=1)).detach()
            samples += velocity * V.delta_t
        x, y = samples[:,0], samples[:,1]
        x_array.append(x.numpy())
        y_array.append(y.numpy())
        plt.figure(figsize=(10, 5))
        plt.suptitle(f'Plots for delta t={delta_t}')
        plt.scatter(x_array, y_array, c='blue', alpha=0.5)
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.tight_layout()
        plt.show()

def Q5():
    global t
    points_not_from_rings = [np.array([1, 1]), np.array([-1, -1])]
    points_from_rings = [np.array([1, 0.5]), np.array([0.5, 1]), np.array([0.5, -1])]
    points = np.vstack((points_from_rings, points_not_from_rings)) ##chatGPT
    points = torch.tensor(points, dtype=torch.float32)##chatGPT
    plot_scatter([points[:, 0].numpy()], [points[:, 1].numpy()], label_prefix='Initial')
    all_trajectories=[]
    for t in np.arange(0, 1 + V.delta_t, V.delta_t):
        v_t = V(points, torch.tensor(t, dtype=torch.float32).repeat(5, 1)).detach()
        points = points - v_t * V.delta_t
        all_trajectories.append(points.numpy().copy())
    all_trajectories = np.array(all_trajectories)
    plot_trajectory(all_trajectories, 5)

    all_trajectories=[]
    for t in np.arange(0, 1 + V.delta_t, V.delta_t):
        v_t = V(points, torch.tensor(t, dtype=torch.float32).repeat(5, 1)).detach()
        points = points + v_t * V.delta_t
        all_trajectories.append(points.numpy().copy())
    all_trajectories = np.array(all_trajectories)
    plot_trajectory(all_trajectories, 5)

def plot_scatter(x_array, y_array, label_prefix='Forward'): ##ChatGPT
    plt.figure(figsize=(10, 6))
    for i, (x, y) in enumerate(zip(x_array, y_array)):
        plt.scatter(x, y, label=f'{label_prefix} t={i}')
        for j in range(len(x)):
            plt.text(x[j], y[j], f'{j}', fontsize=9)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'{label_prefix} Trajectories')
    plt.show()

Q1(loss_array)
Q2()
Q3()
Q4()
Q5()
