import tensorflow as tf
import os
from PIL import Image
import numpy as np

def save_checkpoint(manager):
    save_path = manager.save()
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path, model, optimizer, step_variable):
    try:
        dummy_images = tf.zeros((1, 299, 299, 3), dtype=tf.float32)
        dummy_captions = tf.zeros((1, 10), dtype=tf.int64)
        # Call model with dummy input to build it, ensuring training=False for inference-like setup
        _ = model(dummy_images, dummy_captions, training=False) 
        print("Model built with placeholder input.")
    except Exception as e:
        print(f"Model already built or error during placeholder call: {e}")

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step_variable)
    
    status = checkpoint.restore(checkpoint_path)
    
    status.expect_partial().assert_existing_objects_matched()
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    return step_variable.numpy()

# Removed @tf.no_gradient() decorator
def print_examples(model, dataset, transform, device=None):
    test_img_paths = {
        "dog": "test_examples/dog.jpg",
        "boat": "test_examples/boat.jpg",
        "child": "test_examples/child.jpg"
    }

    print("\n--- Generating Example Captions ---")
    
    # Removed manual model.trainable toggling.
    # The training=False in model.py's caption_image should handle inference mode correctly.

    for name, path in test_img_paths.items():
        if not os.path.exists(path):
            print(f"Warning: Test image '{path}' not found. Skipping example for {name}.")
            continue
        try:
            pil_img = Image.open(path).convert("RGB")
            img_tensor = transform(pil_img)
            
            # The `caption_image` method in model.py internally handles `training=False`
            caption_list = model.caption_image(img_tensor, dataset.vocab)
            
            print(f"Example: {name.capitalize()} image")
            print(f"Generated Caption: {' '.join(caption_list)}")
        except Exception as e:
            print(f"Error processing example for {name}: {e}")
    print("-----------------------------------")
