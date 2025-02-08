import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, models
import numpy as np

# --------------------------------------------------------------------------------
# Squeeze-and-Excitation (SE) block
# --------------------------------------------------------------------------------
def squeeze_excitation_block(input_tensor, reduction_ratio=16):
    """
    A simple Squeeze-and-Excitation (SE) block:
      1. Squeeze = GlobalAveragePooling2D -> shape: [batch, channels]
      2. Excitation = 2 small dense layers to learn channel importance -> shape: [batch, channels]
      3. Reshape to [batch, 1, 1, channels]
      4. Multiply with original input tensor (broadcasted across H,W)
    """
    channel_dim = int(input_tensor.shape[-1])

    # Squeeze
    se = layers.GlobalAveragePooling2D()(input_tensor)  # [batch, channels]

    # Excitation
    se = layers.Dense(channel_dim // reduction_ratio, activation='relu')(se)
    se = layers.Dense(channel_dim, activation='sigmoid')(se)

    # Reshape to [batch, 1, 1, channels]
    se = layers.Reshape((1, 1, channel_dim))(se)

    # Scale (broadcasting automatically matches [batch, H, W, channels])
    return layers.Multiply()([input_tensor, se])

# --------------------------------------------------------------------------------
# Depthwise-Separable convolution utility
# --------------------------------------------------------------------------------
def depthwise_separable_conv(x, filters, kernel_size, strides=(1, 1), padding='same'):
    """A Depthwise-Separable conv block with BN + ReLU."""
    x = layers.SeparableConv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

# --------------------------------------------------------------------------------
# Pretrained EfficientNetV2B0 Bottleneck
# --------------------------------------------------------------------------------
def bottleneck0(inputs):
    """Branch A: EfficientNetV2B0 as a global feature extractor (frozen by default)."""
    backbone = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False)

    # Freeze entire backbone to reduce computation
    for layer in backbone.layers:
        layer.trainable = False

    x = backbone(inputs)                            # [batch, h, w, channels]
    x = layers.GlobalAveragePooling2D()(x)          # [batch, channels]
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)     # Final bottleneck
    return x

# --------------------------------------------------------------------------------
# Neck Section
# --------------------------------------------------------------------------------
def neck_section(inputs):
    """Branch C: One depthwise-separable conv + max-pool + optional SE block, then flatten."""
    x = depthwise_separable_conv(inputs, filters=256, kernel_size=7, strides=(2, 2), padding='same')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Squeeze-and-Excitation
    x = squeeze_excitation_block(x)

    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Vision Transform (patch-based) branch
# --------------------------------------------------------------------------------
def vision_transform(inputs, patch_size):
    """
    Branch B:
      - Reshape to 'patch_size x patch_size x ?'
      - Normalize
      - Small CNN with depthwise-separable conv + max-pooling + SE
      - Flatten
    """
    # Reshape the input into a single patch of size `patch_size`.
    x = layers.Reshape((patch_size[0], patch_size[1], -1))(inputs)

    # Normalize pixel values
    x = layers.Lambda(lambda image: K.cast(image, 'float32') / 255.0)(x)

    # Patch-based CNN
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = depthwise_separable_conv(x, 64, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)

    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = depthwise_separable_conv(x, 128, kernel_size=3)
    x = layers.MaxPooling2D((2, 2))(x)

    # Squeeze-and-Excitation
    x = squeeze_excitation_block(x)

    x = layers.Flatten()(x)
    return x

# --------------------------------------------------------------------------------
# Adaptive Branch Fusion
# --------------------------------------------------------------------------------
def adaptive_branch_fusion(*branches, hidden_dim=128):
    """
    - For each branch, learn a scalar gate in [0,1] via a small MLP
    - Multiply branch by gate
    - Concatenate the gated features
    """
    weighted_branches = []
    for branch in branches:
        gate = layers.Dense(hidden_dim, activation='relu')(branch)
        gate = layers.Dense(1, activation='sigmoid')(gate)  # scalar gating factor
        weighted = layers.Multiply()([branch, gate])
        weighted_branches.append(weighted)

    return layers.Concatenate()(weighted_branches)

# --------------------------------------------------------------------------------
# Build the Full Hybrid Model
# --------------------------------------------------------------------------------
def build_hybrid_model(input_shape, num_classes, patch_size):
    """
    Overall architecture:
      - Branch A: EfficientNetV2B0 (frozen) -> Dense(256)
      - Branch B: Patch-based 'vision transform' -> Flatten
      - Branch C: Neck -> Flatten
      - Adaptive gating fusion -> Dense(num_classes) -> Softmax
    """
    inputs = layers.Input(shape=input_shape)

    # Branch A
    efficientnet_features = bottleneck0(inputs)

    # Branch B
    patches_features = vision_transform(inputs, patch_size)

    # Branch C
    neck_features = neck_section(inputs)

    # Fusion
    fused = adaptive_branch_fusion(efficientnet_features, patches_features, neck_features, hidden_dim=64)

    # Final classification
    outputs = layers.Dense(num_classes, activation='softmax')(fused)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":

    # Hyperparameters / shape
    size = 128            
    patch_size = (32, 32) 
    num_classes = 2       # binary classification
    batch_size = 8
    epochs = 50            

    # Build the model under a MirroredStrategy to efficently use multi-GPU 
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
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

    # Print a summary 
    model.summary()

    # Create EarlyStopping Callback
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",      
        patience=10,             
        restore_best_weights=True 
    )

    # Train
    history = model.fit(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_cb]
    )

    # Save the model
    model.save("final_hybrid_model.h5")
    print("Model saved as 'final_hybrid_model.h5'")
