import os
import sys
import json
import time
import warnings

import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit these paths to match your setup
# ═══════════════════════════════════════════════════════════════════════

DATASET_PATH = r"C:\Users\yasin\SLCB\Datasets\data\dataset_static"

# Output directory (same folder as this script)
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# Model outputs
MODEL_PATH   = os.path.join(OUTPUT_DIR, "landmark_model.keras")
LABELS_PATH  = os.path.join(OUTPUT_DIR, "labels.npy")
CONFIG_PATH  = os.path.join(OUTPUT_DIR, "training_config.json")

# Training hyperparameters
TEST_SIZE    = 0.2       # 20% validation split
RANDOM_SEED  = 42
EPOCHS       = 50
BATCH_SIZE   = 64
PATIENCE     = 10        # early stopping patience


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Discover classes
# ═══════════════════════════════════════════════════════════════════════

def discover_classes(dataset_path: str) -> list:
    """Find all class subdirectories in the dataset."""
    classes = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    return classes


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Extract landmarks from images
# ═══════════════════════════════════════════════════════════════════════

NUM_LANDMARKS = 21   # MediaPipe hand model has 21 landmarks
NUM_FEATURES  = NUM_LANDMARKS * 3  # x, y, z per landmark = 63

def normalize_landmarks(landmarks: list) -> np.ndarray:
    """
    Normalize hand landmarks to be translation and scale invariant.

    Process:
      1. Subtract wrist position (landmark 0) → translation invariance
      2. Divide by max distance from wrist → scale invariance
      3. Result: 63 float values centered at origin, scaled to [-1, 1]

    This ensures the same hand pose produces the same values regardless
    of where the hand is in the frame or how far from the camera.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # (21, 3)

    # Translate: center on wrist (landmark 0)
    wrist = coords[0].copy()
    coords -= wrist

    # Flatten Z to 0. MediaPipe's Z axis varies heavily between library versions. 
    # Static signs rely primarily on 2D projection, so we zero out Z to guarantee 
    # alignment with the React frontend's legacy API.
    coords[:, 2] = 0.0

    # Scale: normalize by max distance from wrist (now purely 2D)
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist

    return coords.flatten()  # → 63 floats (where every 3rd float is 0)


def extract_landmarks_from_dataset(dataset_path: str, class_names: list):
    """
    Run MediaPipe Hands on every image in the dataset using the new Tasks API.
    Returns (X, y) arrays of normalized landmarks and class indices.
    """
    # Initialize the HandLandmarker task
    base_options = mp.tasks.BaseOptions(
        model_asset_path=os.path.join(OUTPUT_DIR, "hand_landmarker.task")
    )
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        all_landmarks = []
        all_labels = []
        skipped = 0
        total = 0

        label_map = {name: i for i, name in enumerate(class_names)}
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')

        print("=" * 60)
        print("STEP 2 — Extracting landmarks from dataset images")
        print("=" * 60)

        for cls_name in class_names:
            cls_dir = os.path.join(dataset_path, cls_name)
            cls_idx = label_map[cls_name]
            cls_count = 0
            cls_skip = 0

            image_files = [
                f for f in os.listdir(cls_dir)
                if f.lower().endswith(valid_ext)
            ]

            for fname in image_files:
                total += 1
                img_path = os.path.join(cls_dir, fname)
                
                # Load image via MediaPipe's built-in format
                mp_image = mp.Image.create_from_file(img_path)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks and len(result.hand_landmarks) > 0:
                    hand_lm = result.hand_landmarks[0]
                    features = normalize_landmarks(hand_lm)
                    all_landmarks.append(features)
                    all_labels.append(cls_idx)
                    cls_count += 1
                else:
                    cls_skip += 1
                    skipped += 1

            pct = cls_count / max(len(image_files), 1) * 100
            print(f"  {cls_name:>3}: {cls_count:>5} / {len(image_files):>5} extracted ({pct:.0f}%)"
                  + (f"  ⚠ {cls_skip} skipped" if cls_skip > 0 else ""))

    X = np.array(all_landmarks, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)

    print(f"\n  Total: {len(X)} landmarks extracted from {total} images")
    print(f"  Skipped: {skipped} images (no hand detected)")
    print(f"  Success rate: {len(X)/max(total,1)*100:.1f}%")

    return X, y


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Build and train the model
# ═══════════════════════════════════════════════════════════════════════

def build_model(num_classes: int) -> "tf.keras.Model":
    """
    Build a lightweight Dense classifier for 63-dimensional landmark input.

    Architecture:
      Input(63) → Dense(256) → BN → Dropout(0.3)
                → Dense(128) → BN → Dropout(0.3)
                → Dense(64)  → BN → Dropout(0.3)
                → Dense(num_classes, softmax)

    Total: ~30K params — trains in seconds on CPU.
    """
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(NUM_FEATURES,), name="landmark_input"),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="isl_landmark_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(X_train, y_train, X_val, y_val, num_classes):
    """Train the model with early stopping and LR reduction."""
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("STEP 4 — Training landmark classifier")
    print("=" * 60)

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(f"\n  Train samples: {len(X_train)}")
    print(f"  Val samples:   {len(X_val)}")
    print(f"  Classes:       {num_classes}")
    print(f"  Epochs:        {EPOCHS} (with early stopping, patience={PATIENCE})")
    print()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    return model, history


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Evaluate and save
# ═══════════════════════════════════════════════════════════════════════

def evaluate_and_save(model, X_val, y_val, class_names, history):
    """Print classification report and save all artifacts."""
    import tensorflow as tf

    print("\n" + "=" * 60)
    print("STEP 5 — Evaluation & saving")
    print("=" * 60)

    # Classification report
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names))

    val_acc = max(history.history.get("val_accuracy", [0]))
    train_acc = max(history.history.get("accuracy", [0]))

    # Save model
    model.save(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / 1e6
    print(f"✅ Model saved: {MODEL_PATH} ({model_size:.2f} MB)")

    # Save labels
    np.save(LABELS_PATH, np.array(class_names))
    print(f"✅ Labels saved: {LABELS_PATH} ({len(class_names)} classes)")

    # Save config
    config = {
        "model_type": "landmark_classifier",
        "backbone": "dense_network",
        "num_classes": len(class_names),
        "class_names": class_names,
        "label_map": {name: i for i, name in enumerate(class_names)},
        "input_shape": [NUM_FEATURES],
        "input_type": "landmarks",
        "num_landmarks": NUM_LANDMARKS,
        "landmark_dimensions": 3,
        "model_path": MODEL_PATH,
        "confidence_threshold": 0.60,
        "best_train_accuracy": float(train_acc),
        "best_val_accuracy": float(val_acc),
        "normalization": "wrist_centered_scale_normalized",
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    print(f"✅ Config saved: {CONFIG_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — Summary")
    print("=" * 60)
    print(f"  Best training accuracy:   {train_acc:.4f}")
    print(f"  Best validation accuracy: {val_acc:.4f}")
    print(f"  Model parameters:         {model.count_params():,}")
    print(f"  Model size:               {model_size:.2f} MB")
    print(f"\n  Files saved to: {OUTPUT_DIR}")
    print(f"    • {os.path.basename(MODEL_PATH)}")
    print(f"    • {os.path.basename(LABELS_PATH)}")
    print(f"    • {os.path.basename(CONFIG_PATH)}")

    if val_acc >= 0.95:
        print("\n  🎉 Excellent accuracy! Model is ready for deployment.")
    elif val_acc >= 0.85:
        print("\n  ✅ Good accuracy. Model should work well for most signs.")
    elif val_acc >= 0.70:
        print("\n  ⚠️  Moderate accuracy. Consider collecting more diverse data.")
    else:
        print("\n  ❌ Low accuracy. Something may be wrong — check data quality.")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   ISL Landmark-Based Training Pipeline                   ║")
    print("║   Static Signs: A-Z + 1-9                                ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Check dataset exists
    if not os.path.isdir(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("   Please update DATASET_PATH in this script.")
        sys.exit(1)

    # Step 1: Discover classes
    print("=" * 60)
    print("STEP 1 — Discovering classes")
    print("=" * 60)
    class_names = discover_classes(DATASET_PATH)
    print(f"  Found {len(class_names)} classes: {class_names}")

    # Step 2: Extract landmarks
    t0 = time.time()
    X, y = extract_landmarks_from_dataset(DATASET_PATH, class_names)
    extract_time = time.time() - t0
    print(f"\n  Extraction took: {extract_time:.1f}s")

    if len(X) < 100:
        print("❌ Too few landmarks extracted. Check your dataset and MediaPipe.")
        sys.exit(1)

    # Step 3: Split data
    print("\n" + "=" * 60)
    print("STEP 3 — Splitting data")
    print("=" * 60)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val:   {len(X_val)} samples")

    # Step 4: Train
    t0 = time.time()
    model, history = train_model(X_train, y_train, X_val, y_val, len(class_names))
    train_time = time.time() - t0
    print(f"\n  Training took: {train_time:.1f}s")

    # Step 5: Evaluate and save
    evaluate_and_save(model, X_val, y_val, class_names, history)

    print(f"\n  Total pipeline time: {extract_time + train_time:.1f}s")
    print("\n  Next steps:")
    print("    1. Start the backend:  uvicorn main:app --reload --port 8000")
    print("    2. Start the frontend: npm run dev")
    print("    3. Open the app and test with your webcam!")


if __name__ == "__main__":
    main()
