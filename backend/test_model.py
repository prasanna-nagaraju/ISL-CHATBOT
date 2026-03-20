import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

import numpy as np

print("Step 1: TF imported OK")
print(f"Step 2: TF version = {tf.__version__}")

# Check files exist
model_path = "model_v1/model.keras"
labels_path = "labels.npy"

print(f"Step 3: model.keras exists = {os.path.exists(model_path)}")
print(f"Step 4: labels.npy exists  = {os.path.exists(labels_path)}")

print("Step 5: Loading model... (may take 10-30 sec)")
model = tf.keras.models.load_model(model_path, compile=False)
print("Step 6: Model loaded!")

print("Step 7: Running warm-up prediction...")
dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
result = model.predict(dummy, verbose=0)
print(f"Step 8: Prediction OK — output shape: {result.shape}")

labels = np.load(labels_path, allow_pickle=True).tolist()
print(f"Step 9: Labels loaded — {len(labels)} classes: {labels}")

print("\n✅ Everything works! Backend should start fine.")