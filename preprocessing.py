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
    """Lưu ảnh từ tensor"""
    image = tensor.cpu().clone()
    x = image.clamp(0, 255) / 255.
    x = x.view(x.size(0), 224, 224)
    save_image(x, "{}".format(title))

if __name__ == "__main__":
    # Thiết lập đường dẫn
    root_dir = './AFEW-VA/'  # Đường dẫn đến thư mục gốc chứa các thư mục 01, 02, 03
    output_dir = './AFEW-VA/cropped_faces/'  # Đường dẫn để lưu ảnh đã cắt khuôn mặt
    csv_path = './AFEW-VA/train.csv'  # Đường dẫn tệp CSV đầu ra

    # Tạo thư mục lưu ảnh đã cắt nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Khởi tạo MTCNN cho phát hiện khuôn mặt
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False)
    
    # Khởi tạo danh sách để ghi vào CSV
    csv_data = [['subDirectory_filePath', 'valence', 'arousal']]

    # Duyệt qua các thư mục chính 01, 02, 03
    main_folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    
    for main_folder in main_folders:
        main_folder_path = os.path.join(root_dir, main_folder)
        
        # Duyệt qua các thư mục con (001 đến 050) trong từng thư mục chính
        sub_folders = sorted([sf for sf in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, sf))])
        
        for sub_folder in sub_folders:
            sub_folder_path = os.path.join(main_folder_path, sub_folder)
            json_path = os.path.join(sub_folder_path, f"{sub_folder}.json")  # Đường dẫn đến tệp JSON
            
            # Tạo thư mục lưu ảnh cắt khuôn mặt cho từng thư mục con
            cropped_folder = os.path.join(output_dir, main_folder, sub_folder)
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
            
            # Kiểm tra xem tệp JSON có tồn tại không
            if os.path.exists(json_path):
                # Đọc dữ liệu anotate từ JSON
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
                
                # Duyệt qua từng khung hình .png trong thư mục con
                image_files = sorted(glob.glob(os.path.join(sub_folder_path, '*.png')), key=os.path.getmtime)
                
                for image_path in tqdm(image_files, desc=f"Processing {sub_folder}"):
                    # Xử lý tên ảnh và ID khung hình
                    frame_id = os.path.splitext(os.path.basename(image_path))[0]  # Lấy ID khung hình
                    rel_path = os.path.join(main_folder, sub_folder, os.path.basename(image_path))

                    # Cắt khuôn mặt và lưu ảnh đã cắt
                    img = Image.open(image_path)
                    try:
                        img_cropped = mtcnn(img, save_path=os.path.join(cropped_folder, os.path.basename(image_path)))
                    except TypeError:
                        print(f"Could not process image {image_path}")
                    
                    # Lấy valence và arousal từ JSON
                    valence = annotations['frames'].get(frame_id, {}).get('valence', 0)
                    arousal = annotations['frames'].get(frame_id, {}).get('arousal', 0)
                    
                    # Ghi dữ liệu vào danh sách
                    csv_data.append([rel_path, valence, arousal])

    # Lưu dữ liệu anotate vào tệp CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerows(csv_data)

    print(f"CSV file created successfully at: {csv_path}")
