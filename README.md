# Gender Classification Using Keras

## Overview

This project implements a gender classification model using Keras and TensorFlow. The model is trained on a labeled dataset of human faces to classify gender as either male or female.

## Dataset

The dataset used for training and evaluation consists of labeled images of human faces. Each image is associated with a gender label (`male` or `female`). You can access the dataset using the following link:

[Man | Woman Faces](https://www.kaggle.com/datasets/maciejgronczynski/biggest-genderface-recognition-dataset)
[Hijab Woman Faces](https://www.kaggle.com/datasets/mostafaebrahiem/women-faces-with-hijabscientific-use-only)

### Dataset Structure

- **Total Images:** 27279
- **Classes:** `Male`, `Female`
- **Image Format:** JPEG
- **Resolution:** (128x128)
- **Splitting Ratio:** 80% Training, 19.9% Validation, 0.1% Testing

## Model Architecture

The gender classification model is built using a Convolutional Neural Network (CNN) with the following layers:

```python
inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2, 2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.006)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
```

## Training the Model

The model is trained using the dataset with data augmentation to improve generalization.

```python

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomBrightness(0.1),
])

train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y)
)
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=validation_dataset
)
```

## Model Evaluation

After training, the model is evaluated on the test dataset to measure its performance.

```python
test_dataset_path = 'model_test_images'

test_dataset = tf.keras.utils.image_dataset_from_directory(
  test_dataset_path,
  labels='inferred',
  label_mode='binary',
  batch_size=5,
  image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
  shuffle=False
)
normalization_layer = tf.keras.layers.Rescaling(1./255)
test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)

model_loss, model_accuracy = model.evaluate(test_dataset)

print(f"Test Loss: {model_loss:.2f}")
print(f"Test Accuracy: {model_accuracy * 100:.2f}%")
```

## Results & Performance

- **Training Accuracy:** 94.99%
- **Validation Accuracy:** 92.57%
- **Test Accuracy:** 96.3%
