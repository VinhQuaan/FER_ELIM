import csv
import random

# Paths to the original CSV file and the output files
original_csv = './AFEW-VA/train.csv'
train_csv = './AFEW-VA/training.csv'
val_csv = './AFEW-VA/validation.csv'

# Data split ratio
train_ratio = 0.8  # 80% for training, 20% for validation

# Read data from the original CSV file
with open(original_csv, 'r') as file:
    reader = list(csv.reader(file))
    header = reader[0]  # Extract the header
    data = reader[1:]  # Extract data excluding the header

# Shuffle data randomly
random.shuffle(data)

# Calculate the number of samples for the training set
train_size = int(len(data) * train_ratio)

# Split data into training and validation sets
train_data = data[:train_size]
val_data = data[train_size:]

# Save the training set to a CSV file
with open(train_csv, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(header)  # Write the header
    writer.writerows(train_data)  # Write the training data

# Save the validation set to a CSV file
with open(val_csv, 'w', newline='') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerow(header)  # Write the header
    writer.writerows(val_data)  # Write the validation data

print(f"Data successfully split! training.csv and validation.csv have been created.")
