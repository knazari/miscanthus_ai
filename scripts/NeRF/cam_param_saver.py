import numpy as np
import json

# Assuming a simple calibration, save to JSON for reuse
focal_length = 800
camera_intrinsics = {
    "focal_length": focal_length,
    "principal_point": [960, 540]  # Example for a 1920x1080 image
}

with open("camera_params.json", "w") as f:
    json.dump(camera_intrinsics, f)
print("Camera calibration parameters saved.")
