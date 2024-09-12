# Miscanthus AI

A brief description of what this project does and who it's for.

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [SAM](#sam)
- [Features](#features)
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


