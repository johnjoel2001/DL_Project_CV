import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import cv2

# Define helper functions

def normalize_map(heatmap):
    """Normalize a heatmap to the [0,1] range."""
    heatmap -= heatmap.min()
    denom = (heatmap.max() - heatmap.min()) + 1e-8
    heatmap /= denom
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.5, cmap='jet'):
    """
    Overlay `heatmap` on `img`.
    """
    colormap = plt.cm.get_cmap(cmap)
    colored_heatmap = colormap(heatmap)[..., :3]
    overlay = (1 - alpha) * img + alpha * colored_heatmap
    overlay = np.clip(overlay, 0, 1)
    return overlay

def integrated_gradients(model, x, baseline=None, steps=50, class_idx=0):
    """
    Computes Integrated Gradients 
    """
    x = tf.cast(x, tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(x)
    else:
        baseline = tf.cast(baseline, tf.float32)

    B, H, W, C = x.shape
    
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps+1), [steps+1, 1, 1, 1, 1])
    x_expanded = tf.expand_dims(x, axis=0)
    baseline_expanded = tf.expand_dims(baseline, axis=0)
    interpolated = baseline_expanded + alphas * (x_expanded - baseline_expanded)

    with tf.GradientTape() as tape:
        interpolated_reshaped = tf.reshape(interpolated, [(steps+1)*B, H, W, C])
        tape.watch(interpolated_reshaped)
        preds = model(interpolated_reshaped)
        preds_for_class = preds[:, class_idx]
        loss = tf.reduce_sum(preds_for_class)
    
    grads = tape.gradient(loss, interpolated_reshaped)
    if grads is None:
        raise ValueError("Gradient is None. Not differentiable.")
    
    grads_reshaped = tf.reshape(grads, [steps+1, B, H, W, C])
    avg_grads = tf.reduce_mean(grads_reshaped[1:], axis=0)
    ig = (x - baseline) * avg_grads
    return ig.numpy()

def preprocess_image(image_path, img_size=(128, 128)):
    """Loads and preprocesses an image."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def main(model_path, image_path, class_idx):
    model = tf.keras.models.load_model(model_path)
    img_batch = preprocess_image(image_path)
    ig_map = integrated_gradients(model, img_batch, class_idx=class_idx)
    
    original_img = img_batch[0]
    normalized_img = (original_img - original_img.min()) / (original_img.max() - original_img.min() + 1e-8)
    
    ig_map_single = ig_map[0]
    ig_map_2d = np.mean(ig_map_single, axis=-1)
    ig_map_norm = normalize_map(ig_map_2d)
    overlay_img = overlay_heatmap(normalized_img, ig_map_norm, alpha=0.5, cmap='jet')
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(normalized_img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(ig_map_norm, cmap='jet')
    plt.title('IG Map (2D Mean)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay_img)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the trained model.")
    parser.add_argument("image_path", type=str, help="Path to the test image.")
    parser.add_argument("--class_idx", type=int, default=1, help="Class index for attribution (default=1).")
    args = parser.parse_args()
    
    main(args.model_path, args.image_path, args.class_idx)
