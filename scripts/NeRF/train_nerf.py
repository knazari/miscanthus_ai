from nerf_model import NeRF
import numpy as np
import torch
import torch.nn.functional as F
import json
from PIL import Image
from torchvision import transforms
import os
import csv
import matplotlib.pyplot as plt

# Load camera parameters
with open("camera_params.json", "r") as f:
    camera_params = json.load(f)
focal_length = camera_params["focal_length"]
principal_point = torch.tensor(camera_params["principal_point"], dtype=torch.float32)

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

def sample_rays(images, poses, batch_size, focal):
    """
    Randomly samples rays and their corresponding ground truth RGB values.

    Args:
        images (torch.Tensor): Ground truth images (N, H, W, 3).
        poses (torch.Tensor): Camera poses (N, 4, 4).
        batch_size (int): Number of rays to sample.
        focal (float): Camera focal length.

    Returns:
        torch.Tensor: Ray origins (batch_size, 3).
        torch.Tensor: Ray directions (batch_size, 3).
        torch.Tensor: Ground truth RGB values (batch_size, 3).
    """
    # Select a random image and pose
    img_idx = np.random.randint(images.shape[0])
    image = images[img_idx]
    pose = poses[img_idx]
    
    H, W = image.shape[:2]
    
    # Sample random pixel coordinates in the image
    i = np.random.randint(0, H, size=(batch_size,))
    j = np.random.randint(0, W, size=(batch_size,))
    
    # Get the ground truth RGB values
    ground_truth_rgb = image[i, j]  # Shape: (batch_size, 3)
    
    # Get the ray directions in the camera frame
    ray_dirs = get_ray_directions(H, W, focal)[i, j]  # Select sampled directions
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)  # Normalize
    ray_dirs = ray_dirs.to('cuda')
    
    # print(torch.norm(pose[:3, 3], dim=-1))
    # Transform ray directions to world space
    ray_dirs_world = torch.sum(ray_dirs[..., None, :] * pose[:3, :3], -1)  # Apply rotation
    ray_origins = pose[:3, 3].expand(ray_dirs_world.shape)  # Origin is the camera position

    return ray_origins, ray_dirs_world, ground_truth_rgb

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
    t_vals = torch.linspace(near, far, num_samples).to('cuda')
    t_vals = t_vals.expand(rays_o.shape[0], num_samples).to('cuda')
    
    # Compute sampled points
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
    delta = t_vals[..., 1:] - t_vals[..., :-1]  # Distance between samples
    delta = torch.cat([delta, torch.Tensor([1e10]).expand(delta[..., :1].shape).to('cuda')], -1)  # Append large delta for last sample
    
    # Calculate transmittance T and alpha for each sample
    alpha = 1.0 - torch.exp(-torch.clamp(density, min=0) * delta)
    alpha = alpha.to('cuda')
    transmittance = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to('cuda'), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * transmittance
    
    # Accumulate RGB
    rgb_rendered = torch.sum(weights[..., None] * rgb, -2)
    return rgb_rendered

def image_pose_batch_generator(folder_path, poses_tensor, batch_size=50):
    """
    Generator that yields batches of images and corresponding poses as PyTorch tensors.
    
    Args:
        folder_path (str): Path to the folder containing images.
        poses_tensor (torch.Tensor): Tensor containing all poses, shape (N, 4, 4).
        batch_size (int): Number of images (and poses) per batch.
    
    Yields:
        torch.Tensor: Batch of images tensor of shape (batch_size, H, W, 3).
        torch.Tensor: Batch of poses tensor of shape (batch_size, 4, 4).
    """
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))])
    num_images = len(image_files)
    
    for i in range(0, num_images, batch_size):
        batch_files = image_files[i:i + batch_size]
        images_np = []
        
        # Load the images for the current batch
        for file in batch_files:
            img = Image.open(file).convert("RGB")
            img_np = np.asarray(img)  # Convert to numpy array
            images_np.append(img_np)
        
        # Stack and normalize the batch of images, then convert to PyTorch tensor
        images_np = np.stack(images_np, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1]
        images_tensor = torch.tensor(images_np)  # Convert to torch tensor

        # Select corresponding batch of poses
        poses_batch = poses_tensor[i:i + batch_size]

        yield images_tensor, poses_batch

def parse_colmap_images(file_path):
    """
    Parse COLMAP images.txt file to extract camera poses.

    Args:
        file_path (str): Path to COLMAP's images.txt file.

    Returns:
        torch.Tensor: Poses tensor of shape (N, 4, 4), where N is the number of images.
    """
    poses = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            # Skip comment lines
            if line.startswith("#"):
                continue
            
            # Split the line into components
            data = line.strip().split()
            if len(data) > 11:
                continue
            
            # COLMAP format for each image line:
            # IMAGE_ID, qw, qx, qy, qz, tx, ty, tz, CAMERA_ID, NAME
            qw, qx, qy, qz = map(float, data[1:5])  # Quaternion components
            tx, ty, tz = map(float, data[5:8])      # Translation components
            
            # Convert quaternion to rotation matrix
            R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
            t = np.array([tx, ty, tz]).reshape((3, 1))  # Translation vector
            
            # Create a 4x4 transformation matrix
            pose = np.hstack((R, t))
            pose = np.vstack((pose, np.array([0, 0, 0, 1])))  # Convert to 4x4
            
            # Add pose to list
            poses.append(pose)
    
    # Convert list of poses to a PyTorch tensor
    poses = np.array(poses)
    camera_positions = np.copy(poses[:, :3, 3])  # Extract camera translations from poses

    mean_position = np.mean(camera_positions, axis=0)
    max_distance = np.max(np.linalg.norm(camera_positions - mean_position, axis=-1))
    normalized_camera_positions = (camera_positions - mean_position) / max_distance
    poses[:, :3, 3] = normalized_camera_positions

    # print(poses.shape)
    plt.plot(poses[:, 0, 3], 'r')
    plt.plot(poses[:, 1, 3], 'b')
    plt.plot(poses[:, 2, 3], 'g')
    # print(np.where(poses[:, 0, 2] > 8000000)[0])
    print(poses[100, :3, :4])
    
    plt.show()
    poses_tensor = torch.tensor(np.array(poses), dtype=torch.float32)
    return poses_tensor

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Converts a quaternion into a 3x3 rotation matrix.

    Args:
        qw, qx, qy, qz (float): Quaternion components.

    Returns:
        np.array: 3x3 rotation matrix.
    """
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
   
    return R


# Initialize NeRF model and optimizer
model = NeRF(input_dim=3, hidden_dim=256, output_dim=4, num_layers=8, num_frequencies=10).to('cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# Training parameters
num_epochs = 150  # Set the number of epochs
num_samples = 128
batch_size = 64

# Paths and data
folder_path = "/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/"
poses = parse_colmap_images("/home/kia/Kiyanoush/Github/miscanthus_ai/data/NeRF/images/sparse_model/images.txt").to('cuda')
epoch_loss_log = []  # List to store epoch loss values
epoch_loss_file_path = "epoch_loss.csv"  # File to save epoch loss values
old_loss = 100


# # Open CSV file for appending epoch loss values
# with open(epoch_loss_file_path, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Epoch", "Average Loss"])  # Write header

#     # Training loop
#     for epoch in range(num_epochs):
#         total_loss = 0.0
#         batch_count = 0
#         total_loss = 0.0  # Reset total loss for each epoch
#         batch_count = 0

#         for images_batch, poses_batch in image_pose_batch_generator(folder_path, poses, batch_size=batch_size):

#             images_batch = images_batch.to('cuda')
#             poses_batch = poses_batch.to('cuda')

#             rays_o, rays_d, ground_truth_rgb = sample_rays(images_batch, poses_batch, batch_size, focal_length)  # Obtain a batch of rays and ground truth RGB values

#             rays_o = rays_o.to('cuda')
#             rays_d = rays_d.to('cuda')
#             ground_truth_rgb = ground_truth_rgb.to('cuda')
            
#             # Sample points along each ray
#             points, t_vals = sample_points(rays_o, rays_d, num_samples)
#             points, t_vals = points.to('cuda'), t_vals.to('cuda')  # Move sampled points to GPU
            
#             # Forward pass through NeRF model
#             points_flat = points.reshape(-1, 3)  # Flatten for model input
#             output = model(points_flat)

#             rgb_pred = output[..., :3].reshape(points.shape[:-1] + (3,)).to('cuda')
#             density_pred = output[..., 3].reshape(points.shape[:-1]).to('cuda')
            
#             # Volume rendering to get final RGB
#             rgb_rendered = volume_rendering(rgb_pred, density_pred, t_vals)
            
#             # Calculate loss for the batch
#             # loss = F.mse_loss(rgb_rendered, ground_truth_rgb)
#             loss = torch.nn.SmoothL1Loss()(rgb_rendered, ground_truth_rgb)
#             total_loss += loss.item()  # Accumulate loss for averaging
#             batch_count += 1

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
           
#         # Calculate average loss for the epoch
#         average_epoch_loss = total_loss / batch_count
#         epoch_loss_log.append((epoch, average_epoch_loss))  # Log epoch loss

#         # Write epoch loss to CSV
#         writer.writerow([epoch, average_epoch_loss])
#         print(f"Epoch {epoch}, Average Loss: {average_epoch_loss}")

#         # Optional: Save a model when loss improves
#         if average_epoch_loss < old_loss:
#             torch.save(model.state_dict(), f"nerf_checkpoint.pth")
#             print("loss decreases -> saved the model")
#             old_loss = average_epoch_loss

#         # Optional: Backup epoch loss log every few epochs
#         if epoch % 5 == 0:
#             with open("epoch_loss_backup.csv", "w", newline="") as backup_f:
#                 backup_writer = csv.writer(backup_f)
#                 backup_writer.writerow(["Epoch", "Average Loss"])
#                 backup_writer.writerows(epoch_loss_log)

