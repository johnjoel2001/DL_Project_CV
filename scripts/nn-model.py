import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
import numpy as np

"""
This hybrid deep learning model is a combination of convolutional networks, EfficentNetV2 and vision transformers.

Model Components:
1. EfficientNetV2B0 Bottleneck (Branch A) - Pretrained feature extractor (frozen), producing global features.
2. Patch-Based Vision Transform (Branch B) - Processes input images into smaller patches, applies a CNN, and extracts detailed features.
3. Neck Section (Branch C) - Uses depthwise-separable convolution and Squeeze-and-Excitation (SE) to enhance feature extraction.
4. Adaptive Branch Fusion - Learns adaptive weighting for each branch, combining them into a final representation.
5. Fully Connected Output Layer - Classifies the image as benign or malignant using softmax activation.

This model is optimized for multi-GPU training and employs early stopping to prevent overfitting.
"""

# --------------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) block
# --------------------------------------------------------------------------------
def squeeze_excitation_block(input_tensor, reduction_ratio=16):
    channel_dim = int(input_tensor.shape[-1])
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channel_dim // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channel_dim, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channel_dim))(se)
    return layers.Multiply()([input_tensor, se])

# --------------------------------------------------------------------------------
# Depthwise-Separable convolution utility
# --------------------------------------------------------------------------------
def depthwise_separable_conv(x, filters, kernel_size, strides=(1, 1), padding='same'):
    x = layers.SeparableConv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# --------------------------------------------------------------------------------
# Pretrained EfficientNetV2B0 Bottleneck
# --------------------------------------------------------------------------------
def bottleneck0(inputs):
    backbone = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False)
    for layer in backbone.layers:
        layer.trainable = False
    x = backbone(inputs)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    return x

# --------------------------------------------------------------------------------
# Neck Section
# --------------------------------------------------------------------------------
def neck_section(inputs):
    x = depthwise_separable_conv(inputs, filters=256, kernel_size=7, strides=(2, 2), padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = squeeze_excitation_block(x)
    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Vision Transform (patch-based) branch
# --------------------------------------------------------------------------------
def vision_transform(inputs, patch_size):
    x = layers.Reshape((patch_size[0], patch_size[1], -1))(inputs)
    x = layers.Lambda(lambda image: K.cast(image, 'float32') / 255.0)(x)
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)
    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)
    x = squeeze_excitation_block(x)
    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Adaptive Branch Fusion
# --------------------------------------------------------------------------------
def adaptive_branch_fusion(*branches, hidden_dim=128):
    weighted_branches = []
    for branch in branches:
        gate = layers.Dense(hidden_dim, activation='relu')(branch)
        gate = layers.Dense(1, activation='sigmoid')(gate)
        weighted = layers.Multiply()([branch, gate])
        weighted_branches.append(weighted)
    return layers.Concatenate()(weighted_branches)

# --------------------------------------------------------------------------------
# Build the Full Hybrid Model
# --------------------------------------------------------------------------------
def build_hybrid_model(input_shape, num_classes, patch_size):
    inputs = layers.Input(shape=input_shape)
    efficientnet_features = bottleneck0(inputs)
    patches_features = vision_transform(inputs, patch_size)
    neck_features = neck_section(inputs)
    fused = adaptive_branch_fusion(efficientnet_features, patches_features, neck_features, hidden_dim=64)
    outputs = layers.Dense(num_classes, activation='softmax')(fused)
    return models.Model(inputs=inputs, outputs=outputs)

def train_hybrid_model(X_train, Y_train, X_val, Y_val, size=128, patch_size=(32, 32), num_classes=2, batch_size=8, epochs=50):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = build_hybrid_model(
            input_shape=(size, size, 3),
            num_classes=num_classes,
            patch_size=patch_size
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=["accuracy", tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
        )
    
    model.summary()
    
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",      
        patience=10,             
        restore_best_weights=True 
    )
    
    history = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_cb]
    )
    
    model.save("final_hybrid_model.h5")
    print("Model saved as 'final_hybrid_model.h5'")
