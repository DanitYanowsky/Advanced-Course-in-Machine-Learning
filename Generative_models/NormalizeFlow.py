import torch
import torch.nn as nn
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from torch import optim
from torch.distributions import MultivariateNormal
import numpy as np
from create_data import sample_olympic_rings, create_unconditional_olympic_rings
class AffineCouplingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AffineCouplingLayer, self).__init__()
        self.nn_log_s = NN(hidden_dim=8, input_dim=input_dim//2, output_dim=output_dim//2)
        self.nn_b = NN(hidden_dim=8, input_dim=input_dim//2, output_dim=output_dim//2)

    def forward(self, z_l, z_r):
        log_s, b = self.nn_log_s(z_l), self.nn_b(z_l)
        s= torch.exp(log_s)
        y_r = s * z_r + b
        concatenated = torch.cat((z_l, y_r), dim=-1)
        return concatenated, self.log_det(z_l)

    def inverse(self, y_l, y_r):

        log_s, b = self.nn_log_s(y_l), self.nn_b(y_l)
        s_hat = torch.exp(log_s)
        return torch.cat((y_l, (y_r - b) / s_hat), dim=-1), self.log_det(y_l)

    def log_det(self, y_l):
        s = self.nn_log_s(y_l)
        return -torch.sum(s, dim=-1)


class PermutationLayer(nn.Module): ##chatGPT
    def __init__(self, input_dim):
        super(PermutationLayer, self).__init__()
        self.input_dim = input_dim
        self.permutation = torch.randperm(self.input_dim)
        self.inverse_permutation = torch.argsort(self.permutation)

    def forward(self, x):
        return x[:, self.permutation]

    def inverse(self, y):
        return y[:, self.inverse_permutation]

    def log_det_jacobian(self, x):
        return 0

class FlowModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FlowModel, self).__init__()
        self.input_dim = input_dim
        self.num_layers = 15
        self.layers = nn.ModuleList()
        for layer in range(self.num_layers):
            self.layers.append(AffineCouplingLayer(input_dim=input_dim, output_dim=output_dim))
            if layer!=self.num_layers-1:
                self.layers.append(PermutationLayer(input_dim=input_dim))

    def forward(self, x):
        sum=0
        for layer in self.layers:
            if layer.__class__.__name__ == 'PermutationLayer': ##copilot:
                x = layer(x)
            else:
                x, log_det = layer(x[:,0].view(-1,1), x[:,1].view(-1,1))
        return x, sum

    def inverse(self, y):
        sum=0
        for layer in reversed(self.layers):
            if layer.__class__.__name__ == 'PermutationLayer':
                y = layer.inverse(y)
            else:
                y, log_det = layer.inverse(y[:,0].view(-1,1), y[:,1].view(-1,1))
                sum += log_det
        return y, sum

    def log_probability(self, x):
        z, log_det = self.inverse(x)
        log_pz_x = -0.5 * (2 * np.log(2 * np.pi) + (z ** 2).sum(dim=-1))  ##chat GPT
        loss = log_pz_x.sum() + log_det.sum()
        return loss

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_l):
        x = nn.LeakyReLU()(self.fc1(z_l))
        x = nn.LeakyReLU()(self.fc2(x))
        x = nn.LeakyReLU()(self.fc3(x))
        x = nn.LeakyReLU()(self.fc4(x))
        x = self.fc5(x)
        return x


# torch.manual_seed(5)
# np.random.seed(1)
model = FlowModel(input_dim=2, output_dim=2)
optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size=128
epochs=20
number_of_data_points=250000
samples = torch.tensor(create_unconditional_olympic_rings(number_of_data_points), dtype=torch.float32)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

array_log_pz_x = []
array_log_determinant = []
array_validation_loss=[]
model_save_path = 'model_weights.pth'
optimizer_save_path = 'optimizer_state.pth'

def train_model():
    for epoch in range(epochs):
        train_loss = 0.0
        validation_loss = 0.0
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)
        epoch_log_pz_x, epoch_log_determinant = 0, 0
        model.train()
        data = samples[torch.randperm(samples.size(0))] ##shuffle
        for inputs in range(0, len(samples), batch_size):
            optimizer.zero_grad()
            normal = MultivariateNormal(torch.zeros(2), torch.eye(2))
            batch_data = data[inputs:inputs+batch_size]
            z, log_det=model.inverse(batch_data)
            log_pz_x= -0.5 * (2 * np.log(2 * np.pi) + (z ** 2).sum(dim=-1)) ##chat GPT
            loss =  -log_pz_x.sum() -log_det.sum()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()
            epoch_log_pz_x+=log_pz_x.sum().item()
            epoch_log_determinant+=log_det.sum().item()
        array_log_pz_x.append(epoch_log_pz_x/ len(samples))
        array_log_determinant.append(epoch_log_determinant/ len(samples))
        array_validation_loss.append(train_loss/ len(samples))

        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(samples):.4f}, log_determinant: {epoch_log_determinant/ len(samples):.4f}, log_pz_x: {epoch_log_pz_x/ len(samples):.4f}')
    torch.save(model.state_dict(), model_save_path)

    torch.save(optimizer.state_dict(), optimizer_save_path)



def Q1():
    global epochs
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, epochs+1))
    ##chat GPT:
    plt.plot(epochs, array_validation_loss, label='Validation Loss', marker='o')
    plt.plot(epochs, array_log_determinant, label='Log-Determinant', marker='x')
    plt.plot(epochs, array_log_pz_x, label='log(pz(x))', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    plt.title('Validation Loss and Its Components Over Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_trajectory(all_trajectories, num_samples):
    norm = plt.Normalize(vmin=0, vmax=len(all_trajectories) - 1)
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis  # Using a single colormap
    for i in range(num_samples):
        trajectory = all_trajectories[:, i, :]
        time_steps = np.arange(trajectory.shape[0])
        ##ChatGpt:
        for j in range(trajectory.shape[0] - 1):
            plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1],
                     color=cmap(norm(j)), alpha=0.5)
        plt.scatter(trajectory[:, 0], trajectory[:, 1],
                    c=time_steps, cmap=cmap, norm=norm, marker='o', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory of Points Over Time with NN Velocity')
    plt.grid(True)
    plt.colorbar(label='Timestep')
    plt.show()



def sample_1000(model):
    x_dots = []
    y_dots = []
    gaussian = MultivariateNormal(torch.zeros(2), torch.eye(2))
    for i in range(1000):
        dot, _ = model(torch.unsqueeze(gaussian.sample(), dim=0))
        x_dots.append(dot[0][0].item())
        y_dots.append(dot[0][1].item())

    ##chatGPT
    plt.figure(figsize=(8, 6))  # Adjust figure size as needed
    plt.scatter(x_dots, y_dots, s=10, c='blue', alpha=0.5)  # Adjust size (s), color (c), transparency (alpha) as needed
    plt.title('Scatter Plot of Dots')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.grid(True)
    plt.show()

def Q3(model):
    x_dots = []
    y_dots =[]
    chosen_layers = [0,10,14,20,24,28]
    gaussian = MultivariateNormal(torch.zeros(2), torch.eye(2))
    z=gaussian.sample((1000,))
    for i, layer in enumerate(model.layers):
        if isinstance(layer, PermutationLayer):
            z = layer(z)
        else:
            z, _ = layer(z[:,0].view(-1, 1), z[:,1].view(-1, 1))
        if i in chosen_layers:
            x_dots.append(z.detach().numpy()[:,0])
            y_dots.append(z.detach().numpy()[:,1])


    for i in range(len(chosen_layers)):
        plt.figure(figsize=(8, 6))  # Adjust figure size as needed
        plt.scatter(x_dots[i], y_dots[i], s=10, c='blue', alpha=0.5)  # Adjust size (s), color (c), transparency (alpha) as needed
        plt.title('Scatter Plot of Dots in Layer '+str(chosen_layers[i]))
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.grid(True)
        plt.show()



def Q4():
    global t
    num_samples = 10
    x_dots = []
    y_dots =[]
    z = torch.randn(num_samples, 2, dtype=torch.float32)
    for i, layer in enumerate(model.layers):
        if isinstance(layer, PermutationLayer):
            z = layer(z)
            x_dots.append(z.detach().numpy()[:, 0])
            y_dots.append(z.detach().numpy()[:, 1])
        else:
            z, _ = layer(z[:,0].view(-1, 1), z[:,1].view(-1, 1))
    ## used ChatGpt for the plots
    norm = plt.Normalize(vmin=0, vmax=14)
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis

    for i in range(num_samples):
        for j in range(len(x_dots) - 1):
            plt.plot(x_dots[j:j + 2], y_dots[j:j + 2],
                     color=cmap(norm(j)), alpha=0.5)
        c=np.arange(len(x_dots))
        plt.scatter(np.array(x_dots)[:, i], np.array(y_dots)[:, i],
                    c=c, cmap=cmap, norm=norm, marker='o', alpha=0.5)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory of Points Over Time with NN Velocity')
    plt.grid(True)
    plt.colorbar(label='Timestep')
    plt.show()



def Q5():
    global points, cmap, norm, idx, point, z, layer, _, j, c
    points_not_from_rings = [np.array([1, 1]), np.array([-1, -1])]
    points_from_rings = [np.array([1, 0.5]), np.array([0.5, 1]), np.array([0.5, -1])]
    points = np.vstack((points_from_rings, points_not_from_rings))##chatGPT
    points = torch.tensor(points, dtype=torch.float32)
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=model.num_layers)
    plt.figure(figsize=(12, 8))
    all_trajectories = []
    all_trajectories_rings = trajectory_points(all_trajectories, points)
    plot_trajectories(all_trajectories_rings, ['P0', 'P1', 'P2', 'P3','P4'])
    compute_p_z_x(points)
def plot_trajectories(all_trajectories, all_labels): ##chatGPT
    norm = plt.Normalize(vmin=0, vmax=all_trajectories.shape[0] - 1)
    cmap = plt.cm.viridis

    plt.figure(figsize=(10, 6))
    for i in range(all_trajectories.shape[1]):
        trajectory = all_trajectories[:, i, :]
        time_steps = np.arange(trajectory.shape[0])
        for j in range(trajectory.shape[0] - 1):
            plt.plot(trajectory[j:j + 2, 0], trajectory[j:j + 2, 1],
                     color=cmap(norm(j)), alpha=0.5)
        scatter = plt.scatter(trajectory[:, 0], trajectory[:, 1],
                              c=time_steps, cmap=cmap, norm=norm, marker='o', alpha=0.5)
        plt.text(trajectory[-1, 0], trajectory[-1, 1], all_labels[i],
                 fontsize=12, color='black', weight='bold')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Trajectory of Points Over Time with NN Velocity')
    plt.grid(True)
    plt.colorbar(scatter, label='Timestep')
    plt.show()

def trajectory_points(all_trajectories, points_not_from_rings_torch):
    global idx, point, z, layer, _
    for idx, point in enumerate(points_not_from_rings_torch):
        trajectory = [point.numpy()]
        z = point.unsqueeze(0)
        for layer in reversed(model.layers):
            if isinstance(layer, PermutationLayer):
                z = layer.inverse(z)
            else:
                z, _ = layer.inverse(z[:, 0].view(-1, 1), z[:, 1].view(-1, 1))
                trajectory.append(z.squeeze(0).detach().numpy())
        all_trajectories.append(np.array(trajectory))
    all_trajectories = np.array(all_trajectories).transpose(1, 0, 2)
    return all_trajectories


def compute_p_z_x(points):
    with torch.no_grad():
        for idx, point in enumerate(torch.tensor(points, dtype=torch.float32)):
            log_prob = model.log_probability(point.unsqueeze(0))
            print(f'Point {idx} has log_p(x) of {log_prob.item()}')


def Q2():
    sample_1000(model)
    sample_1000(model)
    sample_1000(model)

train_model()
model.load_state_dict(torch.load(model_save_path))
optimizer.load_state_dict(torch.load(optimizer_save_path))
model.eval()

Q1()
Q2()
Q3(model)
Q4()
Q5()