# Detecting Fake Images using<br> Convolutional Neural Networks (CNNs)

## Author
Violet Yousif

# Overview
## Project Summary
This project develops a Convolutional Neural Network (CNN) to detect fake images using deep learning techniques.
The model utilizes transfer learning with EfficientNetV2B0, data augmentation, and fine-tuning strategies to improve classification accuracy.

## Tools & Technologies:
**Programming:** Python<br>
**Deep Learning Frameworks:** TensorFlow, Keras<br>
**Model Architectures:** EfficientNetV2B0, Custom CNN<br>
**Data Handling:** Pandas, NumPy<br>
**Image Processing:** OpenCV, Matplotlib, ImageDataGenerator<br>
**Cloud & Storage:** Google Colab, Google Drive<br>

## Methods & Techniques:<br>
**Convolutional Neural Networks (CNNs)** for image classification<br>
**Transfer Learning & Feature Extraction** with EfficientNetV2B0<br>
**Data Augmentation** (flipping, rotation, zoom, brightness adjustment)<br>
**Fine-Tuning Pretrained Models** for performance improvement<br>
**Loss Function:** Binary Crossentropy<br>
**Optimizers:** Adam, RMSprop<br>
**Performance Evaluation:** Accuracy & Loss Plots, Heatmaps<br>

## Dataset
The dataset contains two categories:<br>
✔ Real Images<br>
✔ Fake Images<br>

### Dataset Structure:
```train/real/``` - Contains real images for training.<br>
```train/fake/``` - Contains fake images for training.<br>
```valid/``` - Contains images for validation.<br>
```test/``` - Contains images for final model testing.<br>
### Data Source:
The dataset is retrieved from Google Drive, with metadata stored in a CSV file (data.csv).

## Project Workflow
### 1. Data Preprocessing & Visualization<br>
- Mounts Google Drive and loads the dataset.<br>
- Displays dataset structure and image samples.<br>
### 2. Image Augmentation & Preprocessing<br>
- Applies transformations (flipping, rotation, zooming, brightness).<br>
- Uses image_dataset_from_directory() for dataset batching.<br>
### 3. Model Development<br>
- Uses EfficientNetV2B0 (pretrained on ImageNet) as the feature extractor.<br>
- Adds Global Average Pooling, Dropout, and a Dense layer with sigmoid activation.<br>
### 4. Model Compilation & Training<br>
- Compiles the model with:<br>
  - Binary Crossentropy loss<br>
  - Adam optimizer (learning rate: 0.001)<br>
  - Accuracy as the evaluation metric<br>
    - Trains the model and monitors validation performance.<br>
### 5. Fine-Tuning & Optimization<br>
- Unfreezes specific layers of the base model for fine-tuning.<br>
- Re-trains the model with a lower learning rate.<br>
### 6. Performance Evaluation<br>
- Plots training and validation accuracy/loss curves.<br>
- Generates heatmaps for accuracy analysis.<br>
- Evaluates the model on the test set and displays predictions.<br>

## Model Performance & Results
- The model achieves mixed accuracy on both training and validation datasets.
  - Inaccurate results may be due to mixed image datasets for training vs. testing.
- Accuracy and loss graphs help monitor _overfitting_ and _underfitting_.
- Sample predictions demonstrate the model’s ability to _differentiate real and fake images_.
#### Training and Validation Accuracy Graph
#### Training and Validation Loss Graph

# ⚙ Installation & Setup
## Requirements
✔ Python 3.x<br>
✔ TensorFlow 2.x<br>
✔ Keras<br>
✔ Matplotlib<br>
✔ Pandas<br>
✔ NumPy<br>
✔ Google Colab (for execution)<br>
## Running the Project
1. Upload the dataset to Google Drive.
2. Run the Jupyter Notebook in Google Colab.
3. Install dependencies using:<br>
   ```pip install tensorflow keras numpy pandas matplotlib```<br>
4. Modify dataset paths in the script (main_folder).
5. Execute code cells sequentially to train and test the model.

## Future Improvements
✔ Update and expand dataset with more real and fake images from consistent data collection.<br>
✔ Experiment with different CNN architectures (ResNet, MobileNet).<br>
✔ Implement hyperparameter tuning for better accuracy.<br>
