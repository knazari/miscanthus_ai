import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nerf_model import NeRF  # Your NeRF model definition


def generate_point_cloud(model, bounds, resolution=128, threshold=0.1):
    """
    Generate a 3D point cloud from a trained NeRF model.

    Args:
        model (torch.nn.Module): Trained NeRF model.
        bounds (tuple): Bounds of the 3D space (xmin, xmax, ymin, ymax, zmin, zmax).
        resolution (int): Number of points sampled along each axis.
        threshold (float): Density threshold to include points in the point cloud.

    Returns:
        np.ndarray: Point cloud of shape (N, 6) with (x, y, z, r, g, b).
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x = torch.linspace(xmin, xmax, resolution)
    y = torch.linspace(ymin, ymax, resolution)
    z = torch.linspace(zmin, zmax, resolution)
    
    # Create a grid of points
    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing="ij")
    points = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3).to(device)

    # Query the model
    with torch.no_grad():
        output = model(points)
        rgb = output[..., :3]  # Extract RGB values
        density = output[..., 3]  # Extract density values

    # Filter points with density above the threshold
    mask = density > threshold
    points_filtered = points[mask]
    rgb_filtered = rgb[mask]

    # Combine positions and colors into a single array
    point_cloud = torch.cat([points_filtered, rgb_filtered], dim=-1).cpu().numpy()
    return point_cloud


def plot_point_cloud_matplotlib(point_cloud):
    """
    Visualize a 3D point cloud using Matplotlib.

    Args:
        point_cloud (np.ndarray): Array of shape (N, 6) with (x, y, z, r, g, b).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract positions and colors
    positions = point_cloud[:, :3]
    colors = point_cloud[:, 3:]  # RGB

    # Normalize colors to [0, 1] range
    colors = np.clip(colors, 0, 1)

    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors,
        s=1  # Point size
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

# Define bounds of the 3D scene (adjust as necessary)
bounds = (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeRF(input_dim=3, hidden_dim=256, output_dim=4, num_layers=8, num_frequencies=10)
model.load_state_dict(torch.load("/home/kia/Kiyanoush/Github/miscanthus_ai/nerf_checkpoint.pth", map_location=device))  # Replace with your checkpoint path
model = model.to(device)
model.eval()

# Generate point cloud
point_cloud = generate_point_cloud(model, bounds, resolution=128, threshold=0.1)

# Visualize using Matplotlib
plot_point_cloud_matplotlib(point_cloud)

# Visualize using Plotly (interactive)
# plot_point_cloud_plotly(point_cloud)
