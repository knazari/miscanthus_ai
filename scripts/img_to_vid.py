import cv2
import os

def create_video_from_images(image_folder, output_video_path, output_size=None, fps=30):
    # Get list of all image files in the folder, sorted by name
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    
    # Ensure the image folder is not empty
    if not images:
        print("No images found in the folder!")
        return

    # If output size isn't specified, determine the largest width and height from images
    if output_size is None:
        max_width, max_height = 0, 0
        for image in images:
            img_path = os.path.join(image_folder, image)
            frame = cv2.imread(img_path)
            h, w, _ = frame.shape
            max_width = max(max_width, w)
            max_height = max(max_height, h)
        output_size = (max_width, max_height)
    else:
        max_width, max_height = output_size

    # Define the video codec and create VideoWriter object
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (max_width, max_height))

    # Iterate through images, resize, and write them to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)

        # Resize the image while maintaining its aspect ratio
        h, w, _ = frame.shape
        print(frame.shape)
        scaling_factor = min(max_width / w, max_height / h)
        new_size = (int(w * scaling_factor), int(h * scaling_factor))
        resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

        # Create a blank canvas and center the resized image on it
        canvas = cv2.copyMakeBorder(
            resized_frame,
            top=(max_height - new_size[1]) // 2,
            bottom=(max_height - new_size[1]) - (max_height - new_size[1]) // 2,
            left=(max_width - new_size[0]) // 2,
            right=(max_width - new_size[0]) - (max_width - new_size[0]) // 2,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # Black padding
        )
        
        # Write the frame to the video
        video.write(canvas)

    # Release the video writer
    video.release()
    print(f"Video saved at: {output_video_path}")

input_path = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_4th_october/plant1_cropped"
output_path = "/home/kiyanoush/miscanthus_ws/src/viper_ros/data/walled_garden_4th_october/plant1_cropped"
# Usage
create_video_from_images(input_path, output_path + "/output_video2.mp4", fps=3)
