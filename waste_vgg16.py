# ============================================================
# 1. IMPORTS AND HYPERPARAMETERS
# ============================================================
import os
import warnings
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Model, Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import roc_curve, roc_auc_score
from collections import Counter

# Set image dimensions, batch size, and other hyperparameters
img_rows, img_cols = 224, 224   # Image size to be fed to the network
batch_size = 64                 # Batch size during training
n_epochs = 5                    # Number of epochs for training
validation_split = 0.2          # Use 20% of data for validation
n_classes = 2                   # Number of classes (Organic and Recyclable)
seed = 10                       # Seed for random number generation

# Define the directory paths for training and testing datasets
train_path = "./DATASET/TRAIN/"
test_path = "./DATASET/TEST/"

# Define the class labels (folder names should match these)
labels = ['O', 'R']

# ============================================================
# 2. DATA PREPROCESSING AND DATA AUGMENTATION
# ============================================================

# Create ImageDataGenerator objects for training and validation with data augmentation.
train_datagen = ImageDataGenerator(
    validation_split = validation_split,
    rescale = 1.0 / 255.0,       # Normalize pixel values
    width_shift_range = 0.1,     # Shift images horizontally by 10%
    height_shift_range = 0.1,    # Shift images vertically by 10%
    horizontal_flip = True       # Enable horizontal flipping
)

# Even though validation set typically should not be augmented,
# here we use similar preprocessing (mainly rescaling) for consistency.
validation_datagen = ImageDataGenerator(
    validation_split = validation_split,
    rescale = 1.0 / 255.0,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True
)

# Create a generator for test data (only rescaling).
test_datagen = ImageDataGenerator(
    rescale = 1.0 / 255.0
)

# Create the training generator. The 'subset' parameter splits the directory based on validation_split.
train_generator = train_datagen.flow_from_directory(
    directory = train_path,
    classes = labels,
    batch_size = batch_size,
    class_mode = 'binary',       # Binary mode since there are 2 classes
    shuffle = True,
    target_size = (img_rows, img_cols),
    subset = 'training',
    seed = seed
)

# Create the validation generator.
validation_generator = validation_datagen.flow_from_directory(
    directory = train_path,
    classes = labels,
    batch_size = batch_size,
    class_mode = 'binary',
    shuffle = True,
    target_size = (img_rows, img_cols),
    subset = 'validation',
    seed = seed
)

# Create the test generator.
test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes = labels,
    batch_size = batch_size,
    class_mode = 'binary',
    shuffle = True,
    target_size = (img_rows, img_cols)
)

# ============================================================
# 3. VISUALIZE SAMPLE IMAGES FROM THE TRAINING DATA
# ============================================================
# Generate a few random indices to select representative images from the training generator.
random_indices = np.random.randint(len(train_generator), size=5)

plt.figure(figsize=(20, 10))
for i, rand_idx in enumerate(random_indices):
    # Get a single batch of images and labels from the generator.
    batch = train_generator[rand_idx]
    images = batch[0]
    batch_labels = batch[1]
    
    # Plot one image per random batch
    plt.subplot(1, 5, i + 1)
    plt.imshow(images[0])
    # Note: Here 0 label is assumed to be 'Organic' and 1 is 'Recyclable'
    plt.title('Label: {}'.format('Organic' if batch_labels[0] == 0 else 'Recyclable'))
    plt.axis('off')

plt.tight_layout()
plt.show()

# ============================================================
# 4. BUILDING THE MODEL WITH VGG16 TRANSFER LEARNING
# ============================================================
# 4.1 Load the pre-trained VGG16 model, excluding its top classifier layers.
vgg = VGG16(
    weights = 'imagenet',
    include_top = False,           # Exclude the fully connected (top) layers
    input_shape = (img_rows, img_cols, 3)
)

# 4.2 Extract the output of the last VGG16 convolutional block and flatten it.
output = vgg.layers[-1].output
output = Flatten(name='flatten')(output)

# Create a "base" model that takes VGG16 input and outputs the flattened tensor.
basemodel = Model(inputs = vgg.input, outputs = output)

# 4.3 Freeze all layers in the base model so they are not updated during training.
for layer in basemodel.layers:
    layer.trainable = False

# 4.4 Construct the final model by appending custom layers to the frozen base model.
model = Sequential()
model.add(basemodel)
# Add a fully connected (dense) layer with 1024 units and ReLU activation.
model.add(Dense(1024, activation='relu'))
# Use dropout for regularization.
model.add(Dropout(0.5))
# Add batch normalization for stability.
model.add(BatchNormalization())
# Add another dense layer with 512 units.
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
# The output layer has 1 unit with sigmoid activation (since we're doing binary classification).
model.add(Dense(1, activation='sigmoid'))

# ============================================================
# 5. COMPILING AND TRAINING THE MODEL
# ============================================================
# Compile the model with 'binary_crossentropy' as the loss function and the Adam optimizer.
model.compile(
    loss = 'binary_crossentropy',
    optimizer = "adam",
    metrics = ['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# Set up early stopping to prevent overfitting.
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 3,
    restore_best_weights = True
)

steps_per_epoch = train_generator.n  
validation_steps = validation_generator.n  

# Fit (train) the model.
history = model.fit(
    train_generator,
    # steps_per_epoch = steps_per_epoch,        # Adjust based on dataset size and batch_size
    epochs = n_epochs,
    validation_data = validation_generator,
    # validation_steps = validation_steps,       # Adjust as needed for validation set
    verbose = 1,
    callbacks = [early_stopping]
)

# Save the training history in a DataFrame for plotting.
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
plt.title('Training and Validation Loss')
plt.show()

# Plot AUC
history_df.loc[:, ['auc', 'val_auc']].plot()
plt.title('Training and Validation AUC')
plt.show()


print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
print("Maximum validation AUC: {:.4f}".format(history_df['val_auc'].max()))

# ============================================================
# 6. MODEL TESTING: PREDICTIONS ON TEST IMAGES
# ============================================================
# In this section, we predict on test images that have been read from disk.
# Adjust file paths according to your test set directories.
IMG_DIM = (img_rows, img_cols)

# Get all file paths for test images in each class folder.
test_files_O = glob.glob(os.path.join(test_path, 'O', '*.jpg'))
test_files_R = glob.glob(os.path.join(test_path, 'R', '*.jpg'))
test_files = test_files_O + test_files_R
test_files = shuffle(test_files)[:50]  # Shuffle and select 50 random test images

# Load images and determine their labels.
test_imgs = []
test_labels = []
for filepath in test_files:
    # Load and resize image
    img_array = img_to_array(load_img(filepath, target_size=IMG_DIM))
    test_imgs.append(img_array)
    
    # Extract label from file path. Here we assume the folder name is the label.
    label = os.path.basename(os.path.dirname(filepath))  # 'O' or 'R' (folder name)
    test_labels.append(label)

# Convert lists to numpy arrays.
test_imgs = np.array(test_imgs)
test_labels = np.array(test_labels)
print(Counter(test_labels))
# Scale pixel values to [0,1].
test_imgs_scaled = test_imgs.astype('float32') / 255.0

# Define conversion functions between class names and numeric representation.
def class2num(label):
    return 0 if label == 'O' else 1

def num2class(num):
    # For binary classification, we use a threshold of 0.5.
    return 'O' if num < 0.5 else 'R'

# Convert true labels to numeric values.
test_labels_enc = np.array([class2num(label) for label in test_labels])

# Get model predictions: outputs probabilities.
predictions = model.predict(test_imgs_scaled, verbose=0)

# Convert probabilities to class labels.
predicted_labels = np.array([num2class(pred) for pred in predictions.flatten()])

# Display a few prediction results.
print("Sample predictions on test images:")
for i in range(5):
    print(f"True label: {test_labels[i]}, Predicted: {predicted_labels[i]}")

# ============================================================
# 7. PREDICT ON A SINGLE CUSTOM IMAGE
# ============================================================
# For demonstration, we pick one image from the test set.
custom_im = test_imgs_scaled[2]  # Select an image from the test set

plt.imshow(custom_im)
plt.axis('off')
plt.title("Custom Input Image")
plt.show()

# Reshape the image so that it has an extra dimension (batch dimension)
custom_im_expanded = custom_im.reshape((1, IMG_DIM[0], IMG_DIM[1], 3))
prediction_custom_im = model.predict(custom_im_expanded, verbose=0)
print(f"Prediction for custom image: {num2class(prediction_custom_im[0][0])}")

# Assuming test_labels_enc contains the true binary labels (0 for 'O' and 1 for 'R')
# and predictions contains the model's probability predictions.
y_pred_prob = predictions.flatten()  # Flatten if needed to one-dimensional array.

# Generate ROC curve data.
fpr, tpr, thresholds = roc_curve(test_labels_enc, y_pred_prob)
roc_auc = roc_auc_score(test_labels_enc, y_pred_prob)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='best')
plt.show()