"""
train_unified.py — ISL Unified Training Script (Letters + Words)
Trains a single, unified Dense classifier on BOTH static letter/number signs
and the CSLTR word-level frames. Handles massive class imbalance via robust
landmark data augmentation (synthetic rotation, scaling, jitter).
"""

import os
import sys
import json
import time
import math
import random
import warnings

import cv2
import numpy as np
import mediapipe as mp

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=UserWarning)

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

STATIC_DATASET_PATH = r"C:\Users\yasin\SLCB\Datasets\data\dataset_static"
WORDS_DATASET_PATH  = r"C:\Users\yasin\SLCB\Datasets\data\isl_csltr\Frames_Word_Level"
OUTPUT_DIR          = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH   = os.path.join(OUTPUT_DIR, "unified_model.keras")
LABELS_PATH  = os.path.join(OUTPUT_DIR, "unified_labels.npy")
CONFIG_PATH  = os.path.join(OUTPUT_DIR, "unified_config.json")
TASK_PATH    = os.path.join(OUTPUT_DIR, "hand_landmarker.task")

# Hyperparams
TARGET_SAMPLES_PER_CLASS = 800  # We will augment minority classes up to this, downsample majority
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
EPOCHS       = 60
BATCH_SIZE   = 64
PATIENCE     = 10


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Discover datasets
# ═══════════════════════════════════════════════════════════════════════

def get_image_files(cls_dir):
    return [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

def discover_datasets():
    print("=" * 60)
    print("STEP 1 — Discovering datasets")
    print("=" * 60)
    
    classes = {}
    
    # Static signs (A-Z, 1-9)
    if os.path.isdir(STATIC_DATASET_PATH):
        static_classes = sorted([d for d in os.listdir(STATIC_DATASET_PATH) if os.path.isdir(os.path.join(STATIC_DATASET_PATH, d))])
        for c in static_classes:
            classes[c] = get_image_files(os.path.join(STATIC_DATASET_PATH, c))
        print(f"  Found {len(static_classes)} static letter/number classes.")
    
    # Word signs
    if os.path.isdir(WORDS_DATASET_PATH):
        word_classes = sorted([d for d in os.listdir(WORDS_DATASET_PATH) if os.path.isdir(os.path.join(WORDS_DATASET_PATH, d))])
        for c in word_classes:
            # Clean up class names (e.g. "HELLO_HI" -> "HELLO_HI")
            clean_c = c.replace(" ", "_").upper()
            # If there's a collision, words take precedence or we merge
            if clean_c not in classes:
                classes[clean_c] = []
            
            files = get_image_files(os.path.join(WORDS_DATASET_PATH, c))
            # Filter files to make sure they actually match the class (some folders have mixed content)
            # The CSLTR dataset prefixes filenames with the word, e.g., "HELP (1).jpg"
            # But let's just take all images in the folder as ground truth for now to maximize data
            classes[clean_c].extend(files)
        print(f"  Found {len(word_classes)} word classes.")
    
    class_names = sorted(list(classes.keys()))
    print(f"\n  Total Unified Classes: {len(class_names)}")
    return classes, class_names


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Extract & Base Normalization
# ═══════════════════════════════════════════════════════════════════════

def _base_normalize(landmarks_3d):
    """Zero-center and extract only X,Y into a 42-element array (21x2)."""
    coords = np.array([[lm.x, lm.y] for lm in landmarks_3d]) # (21, 2)
    wrist = coords[0].copy()
    coords -= wrist
    return coords

def extract_raw_landmarks(classes):
    """Extract unscaled, wrist-centered 2D landmarks for all images."""
    print("\n" + "=" * 60)
    print("STEP 2 — Extracting base landmarks from images")
    print("=" * 60)
    
    base_options = mp.tasks.BaseOptions(model_asset_path=TASK_PATH)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options, num_hands=1,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3
    )
    
    raw_data = {c: [] for c in classes.keys()}
    total, skipped = 0, 0
    t0 = time.time()
    
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for c, files in classes.items():
            for fpath in files:
                total += 1
                try:
                    mp_image = mp.Image.create_from_file(fpath)
                    result = landmarker.detect(mp_image)
                    if result.hand_landmarks and len(result.hand_landmarks) > 0:
                        coords = _base_normalize(result.hand_landmarks[0])
                        raw_data[c].append(coords)
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
            
            pct = len(raw_data[c]) / max(len(files), 1) * 100
            print(f"  {c:>15} : {len(raw_data[c]):>4} / {len(files):>4} extracted ({pct:.0f}%)")
            
    print(f"\n  Extraction took: {time.time() - t0:.1f}s")
    print(f"  Total extracted:  {total - skipped}")
    print(f"  Skipped (no hand): {skipped}")
    return raw_data


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Augmentation & Balancing
# ═══════════════════════════════════════════════════════════════════════

def rotate_2d(coords, angle_deg):
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return np.dot(coords, rot_mat)

def augment_landmarks(coords):
    """Apply random Affine transforms to the 21x2 wrist-centered coords."""
    c = coords.copy()
    
    # 1. Random rotation (-15 to +15 degrees)
    angle = random.uniform(-15.0, 15.0)
    c = rotate_2d(c, angle)
    
    # 2. Random scale distortion (stretch/squash slightly, 0.9 to 1.1)
    scale_x = random.uniform(0.9, 1.1)
    scale_y = random.uniform(0.9, 1.1)
    c[:, 0] *= scale_x
    c[:, 1] *= scale_y
    
    # 3. Random noise jitter
    noise = np.random.normal(0, 0.015, c.shape) # very small noise
    c += noise
    
    # 4. Final scale normalization (max distance from origin = 1.0)
    max_dist = np.max(np.linalg.norm(c, axis=1))
    if max_dist > 0:
        c /= max_dist
        
    return c.flatten().astype(np.float32)

def augment_and_balance(raw_data, class_names):
    """Balance classes by augmenting minority and downsampling majority."""
    print("\n" + "=" * 60)
    print("STEP 3 — Data Augmentation & Class Balancing")
    print("=" * 60)
    
    X, y = [], []
    label_map = {name: i for i, name in enumerate(class_names)}
    
    for c in class_names:
        templates = raw_data[c]
        if not templates:
            print(f"  [!] WARNING: Class '{c}' has NO valid landmarks! Dropping.")
            continue
            
        c_idx = label_map[c]
        count = len(templates)
        
        # We want TARGET_SAMPLES_PER_CLASS per class
        generated = 0
        
        # First, add the original templates (scale normalized)
        for t in templates:
            tmp = t.copy()
            md = np.max(np.linalg.norm(tmp, axis=1))
            if md > 0: tmp /= md
            X.append(tmp.flatten().astype(np.float32))
            y.append(c_idx)
            generated += 1
            if generated >= TARGET_SAMPLES_PER_CLASS:
                break
                
        # If we need more, augment randomly from templates
        while generated < TARGET_SAMPLES_PER_CLASS:
            t = random.choice(templates)
            X.append(augment_landmarks(t))
            y.append(c_idx)
            generated += 1
            
        # Print summary for this class
        aug_count = generated - min(count, TARGET_SAMPLES_PER_CLASS)
        ind_str = "augmented!" if aug_count > 0 else "downsampled"
        print(f"  {c:>15} : Used {min(count, TARGET_SAMPLES_PER_CLASS):>4} originals + {aug_count:>4} {ind_str} -> {generated} total")
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n  Final Dataset Size: {len(X)} samples across {len(class_names)} classes.")
    return X, y


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Build & Train Model
# ═══════════════════════════════════════════════════════════════════════

def build_model(num_classes):
    import tensorflow as tf
    # Uses 42 features (21 landmarks x 2 axes [X, Y])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(42,), name="unified_landmark_input"),
        
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(num_classes, activation="softmax", name="output"),
    ], name="isl_unified_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train(X, y, num_classes):
    import tensorflow as tf
    
    print("\n" + "=" * 60)
    print("STEP 4 — Training Unified Classifier")
    print("=" * 60)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    model = build_model(num_classes)
    
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=PATIENCE, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs, verbose=1
    )
    
    return model, history, X_val, y_val


def main():
    if not os.path.isdir(STATIC_DATASET_PATH) or not os.path.isdir(WORDS_DATASET_PATH):
        print("❌ Dataset directories not found! Please check paths.")
        sys.exit(1)
        
    classes_files, class_names = discover_datasets()
    num_classes = len(class_names)
    
    raw_lm = extract_raw_landmarks(classes_files)
    
    X, y = augment_and_balance(raw_lm, class_names)
    
    model, history, X_val, y_val = train(X, y, num_classes)
    
    # Save & Eval
    print("\n" + "=" * 60)
    print("STEP 5 — Evaluation & Saving")
    print("=" * 60)
    
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_val, y_pred, target_names=class_names))
    
    model.save(MODEL_PATH)
    np.save(LABELS_PATH, np.array(class_names))
    
    config = {
        "model_type": "unified_landmark_classifier",
        "num_classes": num_classes,
        "input_features": 42,
        "normalization": "wrist_centered_scale_normalized_2d",
        "best_val_accuracy": float(max(history.history.get("val_accuracy", [0])))
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"✅ Saved model:  {MODEL_PATH}")
    print(f"✅ Saved labels: {LABELS_PATH}")
    print(f"✅ Saved config: {CONFIG_PATH}")
    print("\nNext, update main.py and ISLChatbot.jsx to use 42 features (ignoring Z) and unified_model.keras!")

if __name__ == "__main__":
    main()
