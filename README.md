# Waste Material Image Classification

This repository contains Python scripts for classifying waste materials into two categories:Organic and Recyclable using deep learning models.

It implements Transfer Learning with three popular architectures:

VGG16

EfficientNetB0

ResNet50

Each model is trained on a custom waste dataset and evaluated using metrics like Accuracy, F1-Score, Precision, AUC, ROC Curve, and Confusion Matrix.

# Project Structure 
```
🔺 vgg16_script.py         # Waste classification using VGG16

🔺 efficientnet_script.py  # Waste classification using EfficientNetB0

🔺 resnet50_script.py      # Waste classification using ResNet50

🔺 DATASET/
│   🔺 TRAIN/
│   │   🔺 O/              # Organic training images
│   │   🔺 R/              # Recyclable training images
│   🔺 TEST/
│       🔺 O/              # Organic test images
│       🔺 R/              # Recyclable test images
🔺 README.md
```
# Requirements

Install the necessary Python libraries using:

`pip install tensorflow keras numpy pandas matplotlib scikit-learn`

If using a GPU for faster training:

`pip install tensorflow-gpu`

# How to Run the Scripts

1. Clone the Repository

`git clone https://github.com/your-username/waste-classification.git`

`cd waste-classification`


2. Prepare the Dataset
```
 DATASET/
├── TRAIN/
│   ├── O/    # Organic training images
│   └── R/    # Recyclable training images
└── TEST/
    ├── O/    # Organic test images
    └── R/    # Recyclable test images 
```
3. Run the Scripts

To train and evaluate using VGG16: `python vgg16_script.py`

To train and evaluate using EfficientNetB0: `python efficientnet_script.py`

To train and evaluate using ResNet50: `python resnet50_script.py`

# Output and Evaluation

Each script will perform the following:

Load and augment training images

Train the model with early stopping to prevent overfitting

Display sample images from the dataset

Plot Training/Validation Loss and AUC curves

Evaluate the model on a separate test set

Display:

Confusion Matrix

Classification Report (Precision, Recall, F1 Score)

ROC Curve with AUC score

Predict and display a few sample test image predictions


#  Key Features

Transfer Learning with VGG16, EfficientNetB0, and ResNet50

Data Augmentation using ImageDataGenerator

Early Stopping based on validation loss

Comprehensive evaluation (Accuracy, Precision, Recall, F1-Score, AUC)

Easy to extend to more waste categories or other image classification tasks