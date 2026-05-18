from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
import os

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "sparse"
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255)

train_data = train_datagen.flow_from_directory(
    "dataset/training_set",
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
)

val_data = train_datagen.flow_from_directory(
    "dataset/training_set",
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
)

model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid') if class_mode == "binary" else Dense(num_classes, activation='softmax')
])

loss_function = "binary_crossentropy" if class_mode == "binary" else "sparse_categorical_crossentropy"
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(train_data.class_indices)
model.fit(train_data, validation_data=val_data, epochs=5)
test_loss, test_accuracy = model.evaluate(val_data)
print(f"Точность модели на валидационных данных: {test_accuracy:.2f}")
model.save("image_classifier.h5")