from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

model = tf.keras.models.load_model('image_classifier.h5')

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)
    print("Raw prediction:", prediction[0])
    if prediction[0] > 0.5:
        predicted_class = 'dog'
    else:
        predicted_class = 'cat'
    print(f"Модель определила:", predicted_class)

    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(predicted_class)
    plt.axis("off")
    plt.show()

predict_image("images.jpg")

