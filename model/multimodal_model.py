import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2

def build_fever_model(input_size=(224, 224)):
    # ── Rama RGB ───────────────────────────────────────
    rgb_input = Input(shape=(*input_size, 3), name='rgb_input')
    base_rgb  = MobileNetV2(input_shape=(*input_size, 3),
                             include_top=False, weights='imagenet')
    base_rgb.trainable = False
    rgb_feat = base_rgb(rgb_input, training=False)
    rgb_feat = layers.GlobalAveragePooling2D()(rgb_feat)
    rgb_feat = layers.Dense(256, activation='relu')(rgb_feat)
    rgb_feat = layers.BatchNormalization()(rgb_feat)

    # ── Rama IR ────────────────────────────────────────
    ir_input = Input(shape=(*input_size, 1), name='ir_input')
    ir_feat  = layers.Conv2D(32, 3, activation='relu', padding='same')(ir_input)
    ir_feat  = layers.BatchNormalization()(ir_feat)
    ir_feat  = layers.MaxPooling2D(2)(ir_feat)
    ir_feat  = layers.Conv2D(64, 3, activation='relu', padding='same')(ir_feat)
    ir_feat  = layers.BatchNormalization()(ir_feat)
    ir_feat  = layers.MaxPooling2D(2)(ir_feat)
    ir_feat  = layers.Conv2D(128, 3, activation='relu', padding='same')(ir_feat)
    ir_feat  = layers.GlobalAveragePooling2D()(ir_feat)
    ir_feat  = layers.Dense(128, activation='relu')(ir_feat)
    ir_feat  = layers.BatchNormalization()(ir_feat)

    # ── Fusión tardía ──────────────────────────────────
    merged = layers.Concatenate(name='fusion')([rgb_feat, ir_feat])
    merged = layers.Dense(256, activation='relu')(merged)
    merged = layers.Dropout(0.4)(merged)
    merged = layers.Dense(128, activation='relu')(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(1, activation='sigmoid',
                           name='fever_prob')(merged)

    model = models.Model(inputs=[rgb_input, ir_input],
                          outputs=output,
                          name='FeverDetector_Multimodal')
    return model