import cv2
import os

# Path to the folder containing images
# image_folder = './Ece740_data/subject02_video1/azure_kinect_0/color'
image_folder = './Ece740_data/subject02_video2/kinect_v2_2/color'

# Get the list of image filenames sorted by name
images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".jpg")]  # Change the extension if needed

# Get the dimensions of the first image to set up the video writer
img = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = img.shape

# Define the output video file name and properties
video_name = 'subject02_video2_kinect_v2_2.mp4'
fps = 15
video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Iterate through images and write to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release video writer and destroy any OpenCV windows
video.release()
cv2.destroyAllWindows()

print(f"Video '{video_name}' created successfully!")
