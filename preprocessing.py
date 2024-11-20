# -*- coding: utf-8 -*-
import os
import csv
import glob
import json
from PIL import Image
import torch
from torchvision.utils import save_image
from facenet_pytorch import MTCNN
from tqdm import tqdm

def save_im(tensor, title):
    """Save image from tensor"""
    image = tensor.cpu().clone()
    x = image.clamp(0, 255) / 255.
    x = x.view(x.size(0), 224, 224)
    save_image(x, "{}".format(title))

if __name__ == "__main__":
    # Set up paths
    root_dir = './AFEW-VA/'  # Path to the root folder containing directories 01, 02, 03
    output_dir = './AFEW-VA/cropped_faces/'  # Path to save cropped face images
    csv_path = './AFEW-VA/train.csv'  # Path to the output CSV file

    # Create directory for saving cropped images if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize MTCNN for face detection
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False)
    
    # Initialize list to write to CSV
    csv_data = [['subDirectory_filePath', 'valence', 'arousal']]

    # Loop through main directories 01, 02, 03
    main_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    
    for main_folder in main_folders:
        main_folder_path = os.path.join(root_dir, main_folder)
        
        # Loop through subdirectories (001 to 050) within each main directory
        sub_folders = sorted([sf for sf in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, sf))])
        
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
            json_path = os.path.join(sub_folder_path, f"{sub_folder}.json")  # Path to JSON file
            
            # Create directory to save cropped face images for each subdirectory
            cropped_folder = os.path.join(output_dir, main_folder, sub_folder)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
            
            # Check if JSON file exists
            if os.path.exists(json_path):
                # Read annotation data from JSON
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
                
                # Loop through each .png frame in the subdirectory
                image_files = sorted(glob.glob(os.path.join(sub_folder_path, '*.png')), key=os.path.getmtime)
                
                for image_path in tqdm(image_files, desc=f"Processing {sub_folder}"):
                    # Process image name and frame ID
                    frame_id = os.path.splitext(os.path.basename(image_path))[0]  # Get frame ID
                    rel_path = os.path.join(main_folder, sub_folder, os.path.basename(image_path))

                    # Crop face and save the cropped image
                    img = Image.open(image_path)
                    try:
                        img_cropped = mtcnn(img, save_path=os.path.join(cropped_folder, os.path.basename(image_path)))
                    except TypeError:
                        print(f"Could not process image {image_path}")
                    
                    # Get valence and arousal from JSON
                    valence = annotations['frames'].get(frame_id, {}).get('valence', 0)
                    arousal = annotations['frames'].get(frame_id, {}).get('arousal', 0)
                    
                    # Append data to the list
                    csv_data.append([rel_path, valence, arousal])

    # Save annotation data to CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerows(csv_data)

    print(f"CSV file created successfully at: {csv_path}")
