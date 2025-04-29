# modified code for directly getting the model in .tflite with increased accuracy and tfl identifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Initial epochs for training
#FINE_TUNE_EPOCHS = 5  # Additional epochs for fine-tuning
NUM_CLASSES = 2  # Healthy and Bleached corals

# Dataset paths
train_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Training"
val_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Validation"
test_dir = "C:\\newprograms\\All_Projects\\Corals_new\\Model\\Dataset\\Testing"

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],  # Adjust brightness
    fill_mode="nearest",
)

# Validation and Test Data (No Augmentation)
val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Load datasets
train_dataset = train_datagen.flow_from_directory(
    train_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
val_dataset = val_test_datagen.flow_from_directory(
    val_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary"
)
test_dataset = val_test_datagen.flow_from_directory(
    test_dir, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode="binary", shuffle=False
)

# Define DenseNet Model
def create_densenet_model():
    # Load pre-trained DenseNet121 model
    base_model = keras.applications.DenseNet121(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    base_model.trainable = False  # Freeze the base model initially
    # Add custom layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train DenseNet Model
densenet_model = create_densenet_model()
history_densenet = densenet_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping]
)

# Save Models as .tflite with TFL Identifier
def save_as_tflite(model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_filename = f"{model_name}_TFL.tflite"  # Add TFL identifier to the filename
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print(f"✅ {model_name} saved as {tflite_filename}")

# Save DenseNet model as .tflite
save_as_tflite(densenet_model, "densenet_model")

# Evaluate Models
def evaluate_model(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Predict classes
    y_pred = model.predict(test_dataset)
    y_pred = np.round(y_pred).flatten()  # Convert probabilities to binary predictions
    
    # True labels
    y_true = test_dataset.classes
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=test_dataset.class_indices.keys()))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

print("DenseNet Model Evaluation:")
evaluate_model(densenet_model, test_dataset)

# Plot Training Results
def plot_results(history, model_name):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], 'r', label='Train Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], 'r', label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    plt.show()
    
plot_results(history_densenet, "DenseNet")
