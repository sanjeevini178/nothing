# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, MobileNetV2
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Path to the dataset (replace with your actual path)
dataset_path = "path/to/your/dataset"

# Set up ImageDataGenerator for data loading with validation split
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load training and validation data
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical',  # Categorical for multi-class classification
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# Detect the number of classes
num_classes = len(train_generator.class_indices)

# Visualize some training samples
def plot_samples(generator):
    x, y = next(generator)
    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(x[i])
        label_index = np.argmax(y[i])  # Get the label index
        label_name = list(generator.class_indices.keys())[label_index]
        plt.title(label_name)
        plt.axis('off')
    plt.show()

plot_samples(train_generator)

# Create a generic function to build and compile the model
def create_model(base_model, num_classes):
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',  # Categorical crossentropy for multi-class
                  metrics=['accuracy'])
    return model

# Load pre-trained models
vgg16_base = VGG16(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
mobilenet_base = MobileNetV2(input_shape=(150, 150, 3), include_top=False, weights='imagenet')

# Freeze base model layers
for layer in vgg16_base.layers:
    layer.trainable = False
for layer in mobilenet_base.layers:
    layer.trainable = False

# Create the models
vgg16_model = create_model(vgg16_base, num_classes)
mobilenet_model = create_model(mobilenet_base, num_classes)

# Train the models with reduced epochs
epochs = 3  # Adjust based on your requirements

print("Training VGG16 model")
vgg16_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

print("Training MobileNetV2 model")
mobilenet_model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Evaluate and plot confusion matrix
def evaluate_model(model, generator, title):
    loss, accuracy = model.evaluate(generator)
    print(f"{title} - Accuracy: {accuracy:.2f}, Loss: {loss:.2f}")

    # Generate predictions and confusion matrix
    y_true = generator.classes
    y_pred = np.argmax(model.predict(generator), axis=1)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    ConfusionMatrixDisplay(conf_matrix, display_labels=generator.class_indices.keys()).plot()
    plt.title(f"{title} Confusion Matrix")
    plt.show()

# Evaluate both models
evaluate_model(vgg16_model, validation_generator, "VGG16 Model")
evaluate_model(mobilenet_model, validation_generator, "MobileNetV2 Model")

# Visualize predictions
def plot_predictions(model, generator):
    x, y_true = next(generator)
    y_pred = np.argmax(model.predict(x), axis=1)
    y_true = np.argmax(y_true, axis=1)

    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(x[i])
        pred_label = list(generator.class_indices.keys())[y_pred[i]]
        true_label = list(generator.class_indices.keys())[y_true[i]]
        plt.title(f"Pred: {pred_label} | True: {true_label}")
        plt.axis('off')
    plt.show()

# Plot predictions for VGG16 and MobileNetV2 models
print("VGG16 Model Predictions")
plot_predictions(vgg16_model, validation_generator)

print("MobileNetV2 Model Predictions")
plot_predictions(mobilenet_model, validation_generator)
