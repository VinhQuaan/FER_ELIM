```markdown
# FER_ELIM Project Guide

## 1. Introduction
This project performs emotion analysis from image data. Below are the steps to set up the environment, preprocess and prepare data, and train the model.

---

## 2. Environment Setup

### Requirements
- Python 3.8 or higher
- CUDA (if you want to use GPU acceleration for training)

### Step 1: Install Dependencies from `requirements.txt`

To install the required libraries, use `pip` to install them from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## 3. Data Preprocessing

### Step 2: Run `preprocessing.py`

After installing the dependencies, run the `preprocessing.py` file to preprocess the image data and save the results to a CSV file:

```bash
python preprocessing.py
```

### Note:
- `preprocessing.py` will automatically load data from the specified directory and save the processed results to a CSV file (`training.csv`) for later use.

---

## 4. Dataset Splitting

### Step 3: Run `split_dataset.py`

Next, run `split_dataset.py` to split the data in `training.csv` into training and validation sets:

```bash
python split_dataset.py
```

### Note:
- The training and validation sets will be saved as `train.csv` and `validation.csv`, preparing them for model training.

---

## 5. Model Training

### Step 4: Run `run.sh`

Finally, you can start training the model by running the `run.sh` script:

```bash
./run.sh
```

The `run.sh` script will execute `main.py` with preset arguments and begin the training process.

---

## 6. Additional Notes

- **GPU**: To use GPU, ensure your device has the appropriate `CUDA` and `cuDNN` versions installed.
- **CUDA_VISIBLE_DEVICES**: Make sure this environment variable is set in `run.sh` to specify the GPU if your machine has multiple GPUs.
- **Data Directory Structure**: Verify that all necessary data files and directories are in the correct locations as expected by the code.

---

## 7. Contact
For questions or issues with the project, please contact the project manager.
```

### Additional Explanation

- **`requirements.txt`**: Lists all the libraries needed to run the project.
- **`preprocessing.py`**: Preprocesses the image data to the required format.
- **`split_dataset.py`**: Splits the dataset into training and validation sets.
- **`run.sh`**: Initiates the model training process with specified arguments.
