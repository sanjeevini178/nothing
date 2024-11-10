# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Path to the dataset
dataset_path = "path"  # Replace with your actual path

# Step i & ii: Load and split the dataset (80% training, 20% validation)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Set to 'categorical' for multi-class
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',  # Set to 'categorical' for multi-class
    subset='validation'
)

# Detect the number of classes automatically
num_classes = len(train_generator.class_indices)

# Step iii: Visualize some training samples with labels
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

# Step iv: Build a custom CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Use softmax for multi-class classification
])

# Step v: Use a categorical cross-entropy loss function and Adam optimizer
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Categorical cross-entropy for multi-class
              metrics=['accuracy'])

# Step vii: Train the model and Step viii: Evaluate loss and accuracy after each epoch
history = model.fit(
    train_generator,
    epochs=5,
    validation_data=validation_generator
)

# Plot training and validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.show()

# Step ix: Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(validation_generator)
print(f"Test Accuracy: {test_accuracy:.2f}, Test Loss: {test_loss:.2f}")

# Step x: Confusion matrix
y_true = validation_generator.classes
y_pred = np.argmax(model.predict(validation_generator), axis=1)

# Generate and plot the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
ConfusionMatrixDisplay(conf_matrix, display_labels=validation_generator.class_indices.keys()).plot()
plt.title("Confusion Matrix")
plt.show()
