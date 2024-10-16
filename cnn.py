import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import matplotlib.pyplot as plt
tf.compat.v1.set_random_seed(0)
from tensorflow import keras
import numpy as np
np.random.seed(0)
import itertools
from keras.preprocessing.image import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from sklearn.metrics import precision_score, accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


print(tf.config.experimental.list_physical_devices())
print(tf.test.is_built_with_cuda())


IMAGE_SIZE = 256
BATCH_SIZE = 16
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Datasets/", shuffle=True, image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE)

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1):
    ds_size = len(ds)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).take(val_size)

    return train_ds, val_ds, test_ds


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)


# layers
resize_and_rescale = tf.keras.Sequential([
    keras.layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)
])


data_augmentation = tf.keras.Sequential([
    keras.layers.experimental.preprocessing.RandomFlip(
        "horizontal_and_vertical"),
    keras.layers.experimental.preprocessing.RandomRotation(0.2),

])


model = keras.Sequential()


model.add(resize_and_rescale)
model.add(data_augmentation)

model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                              padding="same", input_shape=(256, 256, 3)))
model.add(keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(3, 3))

model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(3, 3))

model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.MaxPooling2D(3, 3))

model.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"))

model.add(keras.layers.Conv2D(512, (5, 5), activation="relu", padding="same"))
model.add(keras.layers.Conv2D(512, (5, 5), activation="relu", padding="same"))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1568, activation="relu"))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(39, activation="softmax"))

model.build(input_shape=input_shape)

opt = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])
model.summary()

ep = 20
with tf.device('/GPU:0'):
    history = model.fit_generator(train_ds,
                                  validation_data=val_ds,
                                  verbose=1,
                                  epochs=ep)

history = np.load('saved_models/1/my_history.npy', allow_pickle='TRUE').item()
print(history.keys())
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

scores = model.evaluate(test_ds)
model.save('saved_models/first_model.h5')
