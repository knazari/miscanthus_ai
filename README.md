# Miscanthus AI

A brief description of what this project does and who it's for.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [SAM](#sam)
- [COLMAP](#colmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

Provide a brief overview of your project. Include the problem it solves, the technologies used, and any key highlights.

## Getting Started

Instructions on how to get a copy of the project up and running on your local machine.

### Prerequisites

List any software or dependencies required before installation (e.g., Python version, libraries, Node.js, etc.).

```bash
# Example
pip install requirements.txt

```

## sam
These are the steps to deploy Segment Anything Model (SAM) from meta for automatic semantc segmentation for miscanthus plant. This can be used for other segmentation tasks as well.

Meta AI has released SAM (Segment Anything Model), and it is available as a PyTorch model. SAM can segment objects from an image with minimal prompts like points, boxes, or clicks. For this example, I'll show you how to install and use SAM to segment an image by selecting a plant.

### Step 1: Install Required Dependencies

You'll need segment-anything, PyTorch, TorchVision, and OpenCV. You can install the required dependencies with the following:

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python torch torchvision matplotlib
```
### Step 2: Download Pre-trained SAM Model

You can use the pre-trained model weights provided by Meta. Download the model weights from this link for the vit_h model, which is the highest quality.

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
### Step 3: Code to Test SAM

The following code uses the Segment Anything model to load an image, allow you to click a point, and segment the plant or other objects around the clicked area.
```bash
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

# Load the image
image_path = "path_to_your_image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load the SAM model (using the ViT-H variant)
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Path to the downloaded checkpoint
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
predictor = SamPredictor(sam)

# Convert image to the format required by SAM
predictor.set_image(image_rgb)

# Display the image and allow user to click a point to segment
def on_click(event):
    if event.button == 1:  # Left mouse button
        # Capture the clicked point
        input_point = np.array([[event.xdata, event.ydata]])
        input_label = np.array([1])  # Label: 1 means foreground

        # Perform segmentation using the clicked point as a prompt
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )

        # Display the segmented mask on the image
        mask = masks[0]
        plt.imshow(image_rgb)
        plt.imshow(mask, alpha=0.5, cmap='jet')
        plt.title("Segmented Plant")
        plt.show()

# Plot the image and set up the click event
fig, ax = plt.subplots()
ax.imshow(image_rgb)
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Click on the plant to segment")
plt.show()
```

Key Parts of the Code:

Model Loading: The sam_model_registry allows you to load the SAM model. Here we are using the ViT-H (Vision Transformer with high quality).

Image Input: The image is read with cv2 and then converted to RGB because SAM expects RGB input.

Click-Based Segmentation:
        When you click a point on the image, the click coordinates (input_point) are used as a prompt to SAM to generate a segmentation mask.
        The model predicts the segmentation mask based on the clicked point and shows the result by overlaying the mask on the original image.

Visualization: The matplotlib library is used to display the image and the segmented mask.

## COLMAP
use a pre-built Docker image for COLMAP, which simplifies the setup process. This approach will let you run COLMAP in a container without manually configuring dependencies.

Here’s how to use COLMAP via Docker:

Install Docker if you haven’t already:

```bash
sudo apt update
sudo apt install docker.io
```

Pull a COLMAP Docker image:

```bash
docker pull colmap/colmap
```
To run the docker container with a GUI support we need the xhost package:

```bash
sudo apt install x11-xserver-utils
xhost +local:docker
```
To enable the docker container to use the NVIDIA graphic card, we need to install the NVIDIA Container Toolkit:
```bash
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
```

Run COLMAP in a Docker container:
```bash
docker run -it --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/data \
    colmap/colmap colmap gui
```
This command:

* Adds the --gpus all flag to enable GPU access inside the container.
* Exports the display environment and mounts the X11 socket for GUI access.
* Mounts the current directory to /data for accessing files inside the container.

To stop running a docker container you first list them and end the one with correponding id:
```bash
docker ps
docker stop <container_id_or_name>
```
### Performing feature extraction and matching
Now that you have the COLMAP GUI running in Docker, you can follow these steps to load the images, perform feature extraction and matching, and generate the camera poses.

Step 1: Set Up a New Project

 1. Create a New Project:
 • In the COLMAP GUI, go to File > New Project.
 • Set the Project Folder to the mounted folder where your images are located (for example, /data if you mounted the current directory).
 • Set the Image Folder to the directory containing your extracted frames.
 • Save the project file (e.g., project.colmap).
 2. Set Up Database Path:
 • In the Database field, create a new database file in the same project folder (e.g., /data/database.db).
 • Click Save to finalize your project setup.

Step 2: Feature Extraction

 1. Feature Extraction:
 • Go to Processing > Feature Extraction.
 • In the Feature Extraction dialog:
 • Verify that the Image Folder points to your images folder.
 • Leave the other settings as default.
 • Click Run to start feature extraction. COLMAP will detect features (keypoints) in each image, which are necessary for matching.
 2. Wait for Completion:
 • This step might take some time, depending on the number of images and the complexity of the scene.
 • Once complete, you should see output logs indicating the number of features detected per image.

Step 3: Feature Matching

 1. Exhaustive Feature Matching:
 • Go to Processing > Exhaustive Matching.
 • Click Run to start feature matching.
 • This step will compare features across all image pairs and determine matches, which is crucial for estimating camera poses.

Step 4: Sparse Reconstruction (Mapping)

 1. Sparse Reconstruction:
 • Go to Reconstruction > Start Reconstruction to initiate the sparse reconstruction (also known as mapping).
 • In the Mapper settings, set the Database Path and Image Folder to their respective locations.
 • Set the Output Path to a new folder in your project directory (e.g., /data/sparse).
 • Click Run to start the mapping process.
 2. View the Sparse Model:
 • Once the sparse reconstruction completes, you should see a sparse 3D model of the scene and the estimated camera poses in the 3D viewer.
 • You can navigate around the scene using the mouse to inspect the point cloud and camera locations.

Step 5: Export Camera Poses for NeRF

 1. Export the Poses:
 • Go to File > Export > Export model as… and choose the TEXT format.
 • Set the export path to a new folder (e.g., /data/sparse_model).
 2. Locate images.txt:
 • In the exported sparse_model folder, you’ll find a file named images.txt, which contains the camera poses in a format we can parse for NeRF.

Summary of Files for NeRF

After completing these steps, the key file for NeRF is:
 • images.txt: Contains camera poses for each image (frame).

You’re now ready to parse the images.txt file and convert it into a format suitable for the NeRF pipeline.


