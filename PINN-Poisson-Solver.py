import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Poisson equation: uxx + uyy = f(x, y)
def poisson_equation(x, y):
    return -2 * torch.sin(x) * torch.sin(y)

# Define the exact solution of the Poisson equation: u(x, y) = sin(x) * sin(y)
def exact_solution(x, y):
    return torch.sin(x) * torch.sin(y)

# Generate random points within the domain [0, pi] x [0, pi]
def generate_random_points(num_points):
    x_rand = np.random.uniform(0, np.pi, (num_points, 1))
    y_rand = np.random.uniform(0, np.pi, (num_points, 1))
    return x_rand, y_rand

# Define the Physics-Informed Neural Network (PINN) model with smoother output
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # Define the dense layers with more neurons and smoother activation function
        self.dense1 = nn.Linear(2, 400)  # Increased neurons
        self.dropout1 = nn.Dropout(0.1)  # Reduced dropout
        self.dense2 = nn.Linear(400, 400)
        self.dropout2 = nn.Dropout(0.1)
        self.dense3 = nn.Linear(400, 400)
        self.dropout3 = nn.Dropout(0.1)
        self.dense4 = nn.Linear(400, 400)
        self.dropout4 = nn.Dropout(0.1)
        self.dense5 = nn.Linear(400, 400)
        self.dropout5 = nn.Dropout(0.1)
        self.dense6 = nn.Linear(400, 1)

    # Define the forward pass with a smoother activation function (ReLU or softplus)
    def forward(self, inputs):
        x = torch.tanh(self.dense1(inputs))  # Use tanh for smoothness
        x = torch.tanh(self.dense2(x))
        x = torch.tanh(self.dense3(x))
        x = torch.tanh(self.dense4(x))
        x = torch.tanh(self.dense5(x))
        return self.dense6(x)
    # Calculate the Laplacian of the predicted solution

    def laplacian(self, u, x, y):
        inputs = torch.cat([x, y], dim=1)
        u = self(inputs)
        grad_u = torch.autograd.grad(u.sum(), [x, y], create_graph=True)
        grad_u_x, grad_u_y = grad_u[0], grad_u[1]

        laplacian_u_x = torch.autograd.grad(grad_u_x.sum(), x, create_graph=True)[0]
        laplacian_u_y = torch.autograd.grad(grad_u_y.sum(), y, create_graph=True)[0]

        laplacian_u = laplacian_u_x + laplacian_u_y
        return laplacian_u

# Define the loss function
def loss(model, x, y, pde_coeff=1.0, boundary_coeff=1.0):
    predicted_solution = model(torch.cat([x, y], dim=1))
    poisson_residual = model.laplacian(predicted_solution, x, y) - poisson_equation(x, y)
    boundary_residual = predicted_solution - exact_solution(x, y)
    
    mse_pde_residual = torch.mean(torch.square(poisson_residual))
    mse_boundary_residual = torch.mean(torch.square(boundary_residual))

    weighted_pde_residual = pde_coeff * mse_pde_residual
    weighted_boundary_residual = boundary_coeff * mse_boundary_residual

    return weighted_pde_residual + weighted_boundary_residual

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Set the number of training points
num_points = 2000
x_train = np.random.uniform(0, np.pi, (num_points, 1))
y_train = np.random.uniform(0, np.pi, (num_points, 1))


x_train_tensor = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, requires_grad=True)


model = PINN()


optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
loss_history = []
max_norms = []
l2_norms = []
epochs = 2001

for epoch in range(epochs):
    if epoch % 100 == 0:
        additional_points = 200
        x_rand, y_rand = generate_random_points(additional_points)
        x_train_tensor = torch.cat([x_train_tensor, torch.tensor(x_rand, dtype=torch.float32, requires_grad=True)], dim=0)
        y_train_tensor = torch.cat([y_train_tensor, torch.tensor(y_rand, dtype=torch.float32, requires_grad=True)], dim=0)

    optimizer.zero_grad()
    loss_value = loss(model, x_train_tensor, y_train_tensor)
    loss_value.backward()
    optimizer.step()

    max_grad_norm = max([param.grad.norm() for param in model.parameters()])
    l2_grad_norm = torch.sqrt(sum([param.grad.norm()**2 for param in model.parameters()]))
    loss_history.append(loss_value.item())
    max_norms.append(max_grad_norm.item())
    l2_norms.append(l2_grad_norm.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item()}, Max Norm: {max_grad_norm.item()}, L2 Norm: {l2_grad_norm.item()}")

# Generate test points for visualization
x_test = np.linspace(0, np.pi, 100)
y_test = np.linspace(0, np.pi, 100)
x_test, y_test = np.meshgrid(x_test, y_test)
x_test = x_test.flatten().reshape(-1, 1)
y_test = y_test.flatten().reshape(-1, 1)
inputs_test = torch.tensor(np.concatenate([x_test, y_test], axis=1), dtype=torch.float32)

# Predicted solution from the PINN model
predicted_solution = model(inputs_test).detach().numpy().reshape(100, 100)

# Exact solution for comparison
exact_solution_values = exact_solution(torch.tensor(x_test), torch.tensor(y_test)).detach().numpy().reshape(100, 100)

# Visualization of the predicted solution
plt.contourf(x_test.reshape(100, 100), y_test.reshape(100, 100), predicted_solution, cmap='viridis')
plt.colorbar(label='Predicted')
plt.title('PINN Solution 2D chart to Poisson\'s Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Visualization of the exact solution
plt.contourf(x_test.reshape(100, 100), y_test.reshape(100, 100), exact_solution_values, cmap='viridis')
plt.colorbar(label='Exact')
plt.title('Exact Solution 2D chart to Poisson\'s Equation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 3D Surface plot of the predicted solution
fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x_test.reshape(100, 100), y_test.reshape(100, 100), predicted_solution, cmap='viridis')
ax1.set_title('PINN Solution 3D chart to Poisson\'s Equation')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('Predicted Solution')

# 3D Surface plot of the exact solution
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x_test.reshape(100, 100), y_test.reshape(100, 100), exact_solution_values, cmap='viridis')
ax2.set_title('Exact Solution 3D chart to Poisson\'s Equation')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('Exact Solution')


plt.show()
# Plot gradient norms and loss history
plt.figure(figsize=(8, 5))
plt.plot(max_norms, label='Max Gradient Norm')
plt.plot(l2_norms, label='L2 Gradient Norm')
plt.title('Maximum and L2 Gradient Norm During Training')
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Loss changing')
plt.title('Change Loss Function During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Calculate the error between predicted and exact solutions at test points
error = np.abs(predicted_solution - exact_solution_values)
print(f"Maximum error: {np.max(error)}")