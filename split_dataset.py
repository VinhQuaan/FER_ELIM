import csv
import random

data_path = "<YOUR_DATA_PATH>"

# Đường dẫn tới tệp CSV gốc và các tệp đầu ra
original_csv = './AFEW-VA/train.csv'
train_csv = './AFEW-VA/training.csv'
val_csv = './AFEW-VA/validation.csv'

# Tỷ lệ chia dữ liệu
train_ratio = 0.8  # 80% cho training, 20% cho validation

# Đọc dữ liệu từ tệp CSV gốc
with open(original_csv, 'r') as file:
    reader = list(csv.reader(file))
    header = reader[0]  # Lấy tiêu đề
    data = reader[1:]  # Lấy dữ liệu bỏ qua tiêu đề

# Thêm "AFEW-VA/" vào trước mỗi giá trị trong cột "subDirectory_filePath"
for row in data:
    row[0] = data_path + row[0]

# Xáo trộn dữ liệu ngẫu nhiên
random.shuffle(data)

# Tính số lượng mẫu cho tập huấn luyện
train_size = int(len(data) * train_ratio)

# Chia dữ liệu thành hai tập
train_data = data[:train_size]
val_data = data[train_size:]

# Lưu tập training vào tệp CSV
with open(train_csv, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(header)  # Ghi tiêu đề
    writer.writerows(train_data)  # Ghi dữ liệu

# Lưu tập validation vào tệp CSV
with open(val_csv, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(header)  # Ghi tiêu đề
    writer.writerows(val_data)  # Ghi dữ liệu

print(f"Đã chia dữ liệu thành công! Tệp training.csv và validation.csv đã được tạo.")
