import cv2
import os


# Function to extract frames and write file paths to traindata.txt
def extract_frames(video_path, output_folder, output_file):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    with open(output_file, 'w') as f:  # Open the output file in write mode
        while True:
            ret, frame = cap.read()

            # If there are no more frames, break the loop
            if not ret:
                break
            if frame_count < 3500:
                # Generate a 5-digit frame number using zfill
                frame_filename = os.path.join(output_folder, f"{str(frame_count).zfill(5)}.png")
                cv2.imwrite(frame_filename, frame)

                # Write the frame file path to the output file
                f.write(frame_filename + '\n')

            frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {len(os.listdir(output_folder))} frames to {output_folder} and wrote file paths to {output_file}")




import cv2
import os


# Function to extract frames and write file paths to traindata.txt
def extract_frames1(video_path, output_folder, output_file):
    # Check if the output folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    with open(output_file, 'w') as f:  # Open the output file in write mode
        while True:
            ret, frame = cap.read()

            # If there are no more frames, break the loop
            if not ret:
                break
            if frame_count > 3500:
                # Generate a 5-digit frame number using zfill
                frame_filename = os.path.join(output_folder, f"{str(frame_count).zfill(5)}.png")
                cv2.imwrite(frame_filename, frame)

                # Write the frame file path to the output file
                f.write(frame_filename + '\n')

            frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {len(os.listdir(output_folder))} frames to {output_folder} and wrote file paths to {output_file}")


# Path to the video file
video_path = '/kaggle/input/videofile/AdamKinzinger0.mp4'
# Output folder to save the frames
output_folder = '/kaggle/temp/taming-transformers1/data/singlefacedata'
# Output file to save the frame paths
output_file = '/kaggle/temp/taming-transformers1/data/traindata.txt'
# Extract frames and write file paths
extract_frames(video_path, output_folder, output_file)





# Output file to save the frame paths
output_file = '/kaggle/temp/taming-transformers1/data/testdata.txt'
# Extract frames and write file paths
extract_frames1(video_path, output_folder, output_file)

