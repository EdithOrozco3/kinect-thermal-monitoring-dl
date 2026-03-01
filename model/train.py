

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau, TensorBoard)
from model.multimodal_model import build_fever_model
from config import MODEL_INPUT_SIZE, MODEL_PATH

def train(dataset_dir='dataset'):
    aug = ImageDataGenerator(
        rescale=1./255, rotation_range=15,
        horizontal_flip=True, brightness_range=[0.8, 1.2],
        width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1
    )
    val_gen = ImageDataGenerator(rescale=1./255)

    train_g = aug.flow_from_directory(
        f'{dataset_dir}/train', target_size=MODEL_INPUT_SIZE,
        batch_size=32, class_mode='binary', color_mode='rgb'
    )
    val_g = val_gen.flow_from_directory(
        f'{dataset_dir}/val', target_size=MODEL_INPUT_SIZE,
        batch_size=32, class_mode='binary', color_mode='rgb'
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_auc',
                        save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir='logs/', histogram_freq=1)
    ]

    model = build_fever_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    print('\n=== FASE 1: Clasificador (20 épocas) ===')
    model.fit(train_g, validation_data=val_g,
              epochs=20, callbacks=callbacks)

    print('\n=== FASE 2: Fine-tuning (40 épocas) ===')
    base = model.get_layer('mobilenetv2_1.00_224')
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(train_g, validation_data=val_g,
              epochs=40, callbacks=callbacks)
    print(f'Modelo guardado en: {MODEL_PATH}')

if __name__ == '__main__':
    train()