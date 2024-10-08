import torch
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

# Get the path of the current script
current_dir = os.path.dirname(__file__)
# Construct the relative path to the folder
data_folder = os.path.join(current_dir, '..', 'data')
# Load the image
image_path = data_folder + "/walled_garden_4th_october/001/color_5.png"
# image_path = "/home/kia/Downloads/miscanthus_far_view.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Load the SAM model (using the ViT-H variant)
sam_checkpoint = data_folder + "/sam_vit_h_4b8939.pth"  # Path to the downloaded checkpoint
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

        # Apply the mask to the original image
        segmented_image = np.copy(image_rgb)
        segmented_image[mask == 0] = 0  # Mask out the background (set to black)

        # Save the segmented image
        segmented_image_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(data_folder + "/segmented_miscanthus_image.jpg", segmented_image_bgr)
        print("Segmented image saved as 'segmented_plant_image.jpg'")

        # Show the segmented image
        plt.imshow(segmented_image)
        plt.title("Segmented Plant")
        # plt.imshow(image_rgb)
        # plt.imshow(mask, alpha=0.5, cmap='jet')
        # plt.title("Segmented Plant")
        plt.show()

# Plot the image and set up the click event
fig, ax = plt.subplots()
ax.imshow(image_rgb)
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.title("Click on the plant to segment")
plt.show()
