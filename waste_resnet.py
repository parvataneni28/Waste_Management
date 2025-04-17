#!/usr/bin/env python
# ============================================================
# 1. IMPORTS AND HYPERPARAMETERS
# ============================================================
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score

img_rows, img_cols = 224, 224
batch_size = 64
n_epochs = 5
validation_split = 0.2
n_classes = 2
seed = 10

train_path = "./DATASET/TRAIN/"
test_path = "./DATASET/TEST/"
labels = ['O', 'R']

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=validation_split,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary',
    classes=labels,
    subset='training',
    shuffle=True,
    seed=seed
)

validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary',
    classes=labels,
    subset='validation',
    shuffle=True,
    seed=seed
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='binary',
    classes=labels,
    shuffle=True
)

# ============================================================
# 3. BUILDING THE MODEL
# ============================================================
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# ============================================================
# 4. TRAINING
# ============================================================
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=n_epochs,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# 5. VISUALIZATION
# ============================================================
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(title='Training and Validation Loss')
plt.show()

history_df[['auc', 'val_auc']].plot(title='Training and Validation AUC')
plt.show()

print(f"Minimum val_loss: {history_df['val_loss'].min():.4f}")
print(f"Maximum val_auc: {history_df['val_auc'].max():.4f}")

# ============================================================
# 6. TESTING & PREDICTIONS
# ============================================================
IMG_DIM = (img_rows, img_cols)

test_files_O = glob.glob(os.path.join(test_path, 'O', '*.jpg'))
test_files_R = glob.glob(os.path.join(test_path, 'R', '*.jpg'))
test_files = shuffle(test_files_O + test_files_R)[:50]

test_imgs = []
test_labels = []
for f in test_files:
    test_imgs.append(img_to_array(load_img(f, target_size=IMG_DIM)))
    test_labels.append(os.path.basename(os.path.dirname(f)))

test_imgs = np.array(test_imgs) / 255.0
test_labels = np.array(test_labels)

def class2num(label): return 0 if label == 'O' else 1
def num2class(pred): return 'O' if pred < 0.5 else 'R'

test_labels_enc = np.array([class2num(l) for l in test_labels])
predictions = model.predict(test_imgs).flatten()
predicted_labels = [num2class(p) for p in predictions]

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

print("Sample predictions:")
for i in range(5):
    print(f"True: {test_labels[i]}, Predicted: {predicted_labels[i]}")

# ============================================================
# 7. ROC CURVE
# ============================================================
fpr, tpr, thresholds = roc_curve(test_labels_enc, predictions)
roc_auc = roc_auc_score(test_labels_enc, predictions)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
