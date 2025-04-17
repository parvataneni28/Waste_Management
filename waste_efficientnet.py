#!/usr/bin/env python
# ============================================================
# 1. IMPORTS AND HYPERPARAMETERS
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Sequential
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0  # Using EfficientNetB0 instead of VGG16
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score


# Set image dimensions, batch size, and other hyperparameters
img_rows, img_cols = 224, 224    # Image dimensions to be fed to the network
batch_size = 64                  # Batch size during training
n_epochs = 5                     # Number of epochs for training
validation_split = 0.2           # Use 20% of training data for validation
n_classes = 2                    # Number of classes (Organic and Recyclable)
seed = 10                        # Seed for reproducibility

# Define the directory paths for training and testing datasets
train_path = "./DATASET/TRAIN/"
test_path = "./DATASET/TEST/"

# Define the class labels (folder names should match these)
labels = ['O', 'R']


# ============================================================
# 2. DATA PREPROCESSING AND DATA AUGMENTATION
# ============================================================
# For training & validation, we use augmentation (rescaling is common).
train_datagen = ImageDataGenerator(
    validation_split=validation_split,
    rescale=1.0 / 255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(
    validation_split=validation_split,
    rescale=1.0 / 255.0,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# For test data, we only perform rescaling.
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Create the training generator by splitting the data
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    classes=labels,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    target_size=(img_rows, img_cols),
    subset='training',
    seed=seed
)

# Create the validation generator
validation_generator = validation_datagen.flow_from_directory(
    directory=train_path,
    classes=labels,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    target_size=(img_rows, img_cols),
    subset='validation',
    seed=seed
)

# Create the test generator (not used in prediction below)
test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    classes=labels,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True,
    target_size=(img_rows, img_cols)
)


# ============================================================
# 3. VISUALIZE SAMPLE IMAGES FROM THE TRAINING DATA
# ============================================================
# Select and display 5 representative images from random batches
random_indices = np.random.randint(len(train_generator), size=5)

plt.figure(figsize=(20, 10))
for i, rand_idx in enumerate(random_indices):
    batch = train_generator[rand_idx]      # Get a batch of images and labels
    images = batch[0]
    batch_labels = batch[1]
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[0])
    plt.title('Label: {}'.format('Organic' if batch_labels[0] == 0 else 'Recyclable'))
    plt.axis('off')

plt.tight_layout()
plt.show()


# ============================================================
# 4. BUILDING THE MODEL WITH EFFICIENTNET TRANSFER LEARNING
# ============================================================
# Load the pre-trained EfficientNetB0 model without the top dense layers
effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

# Extract features from the last convolution block and flatten them
output = effnet.layers[-1].output
output = Flatten(name='flatten')(output)
basemodel = Model(inputs=effnet.input, outputs=output)

# Freeze all layers in the base model
for layer in basemodel.layers:
    layer.trainable = False

# Build the final model by appending custom layers similar to your VGG16 implementation
model = Sequential()
model.add(basemodel)
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))   # Binary classification output

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)


# ============================================================
# 5. TRAINING THE MODEL
# ============================================================
# Setup early stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fit (train) the model.
history = model.fit(
    train_generator,
    epochs=n_epochs,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[early_stopping]
)

# Save training history to DataFrame and plot loss and AUC curves
history_df = pd.DataFrame(history.history)
history_df[['loss', 'val_loss']].plot(title='Training and Validation Loss')
plt.show()

history_df[['auc', 'val_auc']].plot(title='Training and Validation AUC')
plt.show()

print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
print("Maximum validation AUC: {:.4f}".format(history_df['val_auc'].max()))


# ============================================================
# 6. MODEL TESTING: PREDICTIONS ON TEST IMAGES
# ============================================================
IMG_DIM = (img_rows, img_cols)  # Define the image dimension for resizing

# Get file paths for test images from both classes and randomly select 50 images
test_files_O = glob.glob(os.path.join(test_path, 'O', '*.jpg'))
test_files_R = glob.glob(os.path.join(test_path, 'R', '*.jpg'))
test_files = test_files_O + test_files_R
test_files = shuffle(test_files)[:50]

# Load the images and extract correct labels using os.path.basename
test_imgs = []
test_labels = []
for filepath in test_files:
    img_array = img_to_array(load_img(filepath, target_size=IMG_DIM))
    test_imgs.append(img_array)
    label = os.path.basename(os.path.dirname(filepath))  # Extract folder name, e.g., "O" or "R"
    test_labels.append(label)

test_imgs = np.array(test_imgs)
test_labels = np.array(test_labels)
print("Label distribution:", Counter(test_labels))

# Scale pixel values to [0, 1]
test_imgs_scaled = test_imgs.astype('float32') / 255.0

# Define label conversion functions
def class2num(label):
    return 0 if label == 'O' else 1

def num2class(num):
    return 'O' if num < 0.5 else 'R'

# Convert the true labels to numeric values
test_labels_enc = np.array([class2num(label) for label in test_labels])

# Obtain predictions (probabilities) from the model on test images
predictions = model.predict(test_imgs_scaled, verbose=0)
predicted_labels = np.array([num2class(pred) for pred in predictions.flatten()])

# Convert probabilities to binary predictions
predicted_binary = (predictions > 0.5).astype(int)

# Confusion matrix
conf_matrix = confusion_matrix(test_labels_enc, predicted_binary)
print("Confusion Matrix:\n", conf_matrix)

# Precision and F1 Score
precision = precision_score(test_labels_enc, predicted_binary)
f1 = f1_score(test_labels_enc, predicted_binary)
print(f"\nPrecision: {precision:.4f}")
print(f"F1 Score : {f1:.4f}")

# Full classification report
report = classification_report(test_labels_enc, predicted_binary, target_names=["Organic", "Recyclable"])
print("\nClassification Report:\n", report)


# Display a few prediction results
print("Sample predictions on test images:")
for i in range(5):
    print(f"True label: {test_labels[i]}, Predicted: {predicted_labels[i]}")


# ============================================================
# 7. PREDICT ON A SINGLE CUSTOM IMAGE
# ============================================================
# Select a single test image and predict
custom_im = test_imgs_scaled[2]
plt.imshow(custom_im)
plt.axis('off')
plt.title("Custom Input Image")
plt.show()

# Reshape and predict on the custom image
custom_im_expanded = custom_im.reshape((1, IMG_DIM[0], IMG_DIM[1], 3))
prediction_custom_im = model.predict(custom_im_expanded, verbose=0)
print(f"Prediction for custom image: {num2class(prediction_custom_im[0][0])}")


# ============================================================
# 8. EVALUATION: ROC CURVE AND AUC
# ============================================================
# Compute the ROC curve and AUC score
y_pred_prob = predictions.flatten()   # Flatten predictions if needed
fpr, tpr, thresholds = roc_curve(test_labels_enc, y_pred_prob)
roc_auc = roc_auc_score(test_labels_enc, y_pred_prob)

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.show()
