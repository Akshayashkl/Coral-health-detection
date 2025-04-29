# modified code for directly getting the model in .tflite with increased accuracy and tfl identifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow_hub as hub  

# Define constants
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10  # Initial epochs for training
FINE_TUNE_EPOCHS = 10  # Additional epochs for fine-tuning
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

# Define CNN Model with Dropout and Batch Normalization
def create_cnn_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Define ResNet50 Model with Fine-Tuning
def create_resnet50_model():
    # Load ResNet50 with pre-trained weights (excluding the top classification layer)
    base_model = keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=(128, 128, 3)
    )
    base_model.trainable = False  # Freeze the base model initially

    # Add custom layers on top of ResNet50
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),  # Dropout to prevent overfitting
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Fine-Tune ResNet50 Model
def fine_tune_resnet50_model(model):
    # Unfreeze the last 10 layers of ResNet50
    for layer in model.layers[-10:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
#model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)  # for cnn 

# Checkpoint for CNN Model
cnn_checkpoint = ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True)

# Checkpoint for ResNet50 Model
resnet_checkpoint = ModelCheckpoint('best_resnet50_model.h5', monitor='val_accuracy', save_best_only=True)


# Train CNN Model
cnn_model = create_cnn_model()
history_cnn = cnn_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping, cnn_checkpoint]
)

# Train ResNet50 Model
resnet50_model = create_resnet50_model()
history_resnet50 = resnet50_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stopping, resnet_checkpoint]
)

# Fine-Tune ResNet50 Model
resnet50_model = fine_tune_resnet50_model(resnet50_model)
history_resnet50_finetune = resnet50_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=[reduce_lr, early_stopping, resnet_checkpoint]
)

# Save Models as .tflite with TFL Identifier
def save_as_tflite(model, model_name):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_filename = f"{model_name}_TFL.tflite"  # Add TFL identifier to the filename
    with open(tflite_filename, "wb") as f:
        f.write(tflite_model)
    print(f"✅ {model_name} saved as {tflite_filename}")

# Load the best CNN model and save as .tflite
best_cnn_model = keras.models.load_model('best_cnn_model.h5')
save_as_tflite(best_cnn_model, "cnn_model")

# Load the best ResNet50 model and save as .tflite
best_resnet50_model = keras.models.load_model('best_resnet50_model.h5')
save_as_tflite(best_resnet50_model, "resnet50_model")


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

print("CNN Model Evaluation:")
evaluate_model(cnn_model, test_dataset)

print("ResNet50 Model Evaluation:")
evaluate_model(resnet50_model, test_dataset)



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

plot_results(history_cnn, "CNN")
plot_results(history_resnet50, "ResNet50")
plot_results(history_resnet50_finetune, "ResNet50 Fine-Tuned")

save_as_tflite(cnn_model, "cnn_model")
save_as_tflite(resnet50_model, "resnet50_model")

    