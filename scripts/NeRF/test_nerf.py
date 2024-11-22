import torch
import numpy as np
from PIL import Image
from nerf_model import NeRF  # Your NeRF model definition
import matplotlib.pyplot as plt

# Define helper functions for ray sampling, volume rendering, and saving outputs

def get_ray_directions(H, W, focal):
    """
    Generates ray directions for all pixels in a given image size.
    Args:
        H (int): Image height.
        W (int): Image width.
        focal (float): Focal length of the camera.
    Returns:
        torch.Tensor: Ray directions for each pixel in the camera's frame.
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    return dirs

def sample_points(rays_o, rays_d, num_samples=128, near=0.0, far=1.7):
    """
    Uniformly samples points along rays between near and far planes.
    Args:
        rays_o (torch.Tensor): Origins of rays.
        rays_d (torch.Tensor): Directions of rays.
        num_samples (int): Number of samples per ray.
        near (float): Near plane distance.
        far (float): Far plane distance.
    Returns:
        torch.Tensor: Sampled points along the rays.
    """
    t_vals = torch.linspace(near, far, num_samples)
    t_vals = t_vals.expand(rays_o.shape[0], num_samples).to(device)
    points = rays_o[..., None, :] + rays_d[..., None, :] * t_vals[..., :, None]
    return points, t_vals

def volume_rendering(rgb, density, t_vals):
    """
    Renders colors by accumulating RGB and density values along the ray.
    Args:
        rgb (torch.Tensor): RGB values from the NeRF model for each sampled point.
        density (torch.Tensor): Density values from the NeRF model for each sampled point.
        t_vals (torch.Tensor): Sampled distances along each ray.
    Returns:
        torch.Tensor: Accumulated RGB values along the ray.
    """
    delta = t_vals[..., 1:] - t_vals[..., :-1]
    delta = torch.cat([delta, torch.Tensor([1e10]).expand(delta[..., :1].shape).to(device)], -1)
    alpha = 1.0 - torch.exp(-density * delta)
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    rgb_rendered = torch.sum(weights[..., None] * rgb, -2)
    return rgb_rendered

def render_in_patches(rays_o, rays_d, model, H, W, patch_size=32, num_samples=64):
    rendered_image = torch.zeros((H, W, 3), device=rays_o.device)
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            # Define patch boundaries
            i_end = min(i + patch_size, H)
            j_end = min(j + patch_size, W)
            
            # Extract patch rays
            patch_rays_o = rays_o[i:i_end, j:j_end].reshape(-1, 3)
            patch_rays_d = rays_d[i:i_end, j:j_end].reshape(-1, 3)

            # Sample points along rays
            points, t_vals = sample_points(patch_rays_o, patch_rays_d, num_samples)
            
            # Query model
            points_flat = points.reshape(-1, 3)
            with torch.no_grad():
                output = model(points_flat)
            rgb_pred = output[..., :3].reshape(points.shape[:-1] + (3,))
            density_pred = output[..., 3].reshape(points.shape[:-1])

            # Volume rendering
            rgb_rendered = volume_rendering(rgb_pred, density_pred, t_vals)
            
            # Store in final image
            rendered_image[i:i_end, j:j_end] = rgb_rendered.reshape(i_end - i, j_end - j, 3)
    return rendered_image


# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeRF(input_dim=3, hidden_dim=256, output_dim=4, num_layers=8, num_frequencies=10)
model.load_state_dict(torch.load("/home/kia/Kiyanoush/Github/miscanthus_ai/nerf_checkpoint.pth", map_location=device))  # Replace with your checkpoint path
model = model.to(device)
model.eval()

# Define camera parameters
H, W = 3840, 2160  # Image dimensions
focal = 800.0    # Adjust based on your training camera parameters
camera_pose = torch.tensor([[-0.32970771, -0.22551819, 0.91675208, 0.15260693],  # Identity pose (modify for specific viewpoint)
                            [0.58271625,  0.71539326,  0.38555706, 0.25467522],
                            [-0.74278839, 0.66132747, -0.10445756, -0.05468078],
                            [0, 0, 0, 1]], dtype=torch.float32).to(device)

# Generate rays
ray_dirs = get_ray_directions(H, W, focal).to(device)
ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)  # Normalize
rays_d = torch.sum(ray_dirs[..., None, :] * camera_pose[:3, :3], -1).to(device)
rays_o = camera_pose[:3, 3].expand(rays_d.shape).to(device)

# Render the image in patches
rgb_image = render_in_patches(rays_o, rays_d, model, H, W, patch_size=32).cpu().numpy()

# Save or display the rendered image
plt.imshow(np.clip(rgb_image, 0, 1))  # Clip values to [0, 1] for display
plt.axis('off')
plt.savefig("rendered_image.png", bbox_inches='tight')
plt.show()
