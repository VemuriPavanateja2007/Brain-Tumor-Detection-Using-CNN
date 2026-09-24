import os
import json
import time
import numpy as np
import tensorflow as tf

print("Starting Neuro MRI CNN training and export pipeline...")
tf.random.set_seed(123)
np.random.seed(123)

TRAIN_DIR = '/tmp/repo/Dataset/Training'
TEST_DIR = '/tmp/repo/Dataset/Testing'
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 4

os.makedirs('/app/applet/model', exist_ok=True)
os.makedirs('/app/applet/backend/models', exist_ok=True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=123
).prefetch(tf.data.AUTOTUNE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    seed=123
).prefetch(tf.data.AUTOTUNE)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
print("Classes:", class_names)

# Re-create exact CNN architecture from user repository notebook
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3), name='rescaling'),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv2d'),
    tf.keras.layers.MaxPooling2D(name='max_pooling2d'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2d_1'),
    tf.keras.layers.MaxPooling2D(name='max_pooling2d_1'),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', name='conv2d_2'),
    tf.keras.layers.MaxPooling2D(name='max_pooling2d_2'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(128, activation='relu', name='dense'),
    tf.keras.layers.Dense(len(class_names), activation='softmax', name='dense_1')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

t0 = time.time()
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)
train_time = round(time.time() - t0, 2)
print(f"Training completed in {train_time} seconds.")

# Evaluate model
eval_loss, eval_acc = model.evaluate(val_ds, verbose=1)
print(f"Final Test Evaluation: Accuracy={eval_acc:.4f}, Loss={eval_loss:.4f}")

# Save model formats
keras_path = '/app/applet/model/neuro_mri_cnn.keras'
h5_path = '/app/applet/model/neuro_mri_cnn.h5'
model.save(keras_path)
model.save(h5_path)
print(f"Saved {keras_path} and {h5_path}")

# Copy to backend/models as well
backend_keras = '/app/applet/backend/models/neuro_mri_cnn.keras'
backend_h5 = '/app/applet/backend/models/neuro_mri_cnn.h5'
model.save(backend_keras)
model.save(backend_h5)

# Export weights to JSON for lightweight portable inference in Node and Python
weights_dict = {}
for layer in model.layers:
    w = layer.get_weights()
    if len(w) > 0:
        weights_dict[layer.name] = [param.tolist() for param in w]

weights_path = '/app/applet/model/model_weights.json'
with open(weights_path, 'w') as f:
    json.dump(weights_dict, f)
print(f"Saved portable weights JSON: {weights_path}")

# Save metadata
meta = {
    "model_name": "Neuro MRI CNN",
    "architecture": "Sequential 3-Block CNN (Conv2D -> MaxPool2D x3 -> Dense(128) -> Dense(4, softmax))",
    "classes": class_names,
    "input_shape": [128, 128, 3],
    "input_normalization": "1.0 / 255.0",
    "num_parameters": model.count_params(),
    "training_epochs": EPOCHS,
    "final_eval_accuracy": round(float(eval_acc), 4),
    "final_eval_loss": round(float(eval_loss), 4),
    "history": {
        "accuracy": [round(float(x), 4) for x in history.history['accuracy']],
        "val_accuracy": [round(float(x), 4) for x in history.history['val_accuracy']],
        "loss": [round(float(x), 4) for x in history.history['loss']],
        "val_loss": [round(float(x), 4) for x in history.history['val_loss']]
    }
}

with open('/app/applet/model/model_info.json', 'w') as f:
    json.dump(meta, f, indent=2)

print("Export completed successfully!")
