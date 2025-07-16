import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.summary import create_file_writer, scalar
import numpy as np
import os
from PIL import Image

# Import custom modules
from load_data import get_loader
from model import CNNtoRNN
from model_utils import save_checkpoint, load_checkpoint, print_examples


def train():
    def pil_to_tensor_transform(pil_image):
        img_np = np.array(pil_image)
        img_tensor = tf.convert_to_tensor(img_np, dtype=tf.float32)
        
        img_tensor = tf.image.resize(img_tensor, (356, 356))
        img_tensor = tf.image.random_crop(img_tensor, size=[299, 299, 3])
        
        img_tensor = (img_tensor / 127.5) - 1.0
        return img_tensor

    if not os.path.exists("flickr8k/images") or not os.path.exists("flickr8k/captions.txt"):
        print("Error: 'flickr8k/images' or 'flickr8k/captions.txt' not found.")
        print("Please ensure the dataset is in the correct path.")
        print("Creating placeholder dataset files for demonstration purposes.")
        if not os.path.exists("flickr8k/images"):
            os.makedirs("flickr8k/images")
            Image.new('RGB', (300, 300), color = 'red').save("flickr8k/images/placeholder_image.jpg")
            print("Created placeholder image at flickr8k/images/placeholder_image.jpg")
        if not os.path.exists("flickr8k/captions.txt"):
            with open("flickr8k/captions.txt", "w") as f:
                f.write("image,caption\n")
                f.write("placeholder_image.jpg,A dog is running in the grass.\n")
                f.write("placeholder_image.jpg,A brown dog is playing.\n")
                f.write("placeholder_image.jpg,The dog jumps high.\n")
                f.write("placeholder_image.jpg,A cute dog.\n")
                f.write("placeholder_image.jpg,Happy dog.\n")
            print("Created placeholder_captions.txt")
        if not os.path.exists("flickr8k/images") or not os.path.exists("flickr8k/captions.txt"):
            print("Failed to create placeholder dataset. Exiting.")
            return

    train_loader, dataset = get_loader(
        root_folder="flickr8k/images",
        annotation_file="flickr8k/captions.txt",
        transform=pil_to_tensor_transform,
        batch_size=32,
    )
    
    load_model = False
    save_model = True
    
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab) # Keep this for model initialization
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 10
    
    log_dir = "runs/flickr"
    writer = create_file_writer(log_dir)
    step = tf.Variable(0, dtype=tf.int64) 
    
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)
    
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE 
    )
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step=step)
    manager = tf.train.CheckpointManager(checkpoint, directory="./tf_checkpoints", max_to_keep=3)

    if load_model:
        if manager.latest_checkpoint:
            step.assign(load_checkpoint(manager.latest_checkpoint, model, optimizer, step))
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            print("No checkpoint found, starting training from scratch.")
    
    @tf.function
    def train_step(imgs, captions_input, captions_target):
        with tf.GradientTape() as tape:
            output = model(imgs, captions_input, training=True)
            
            # --- FIX STARTS HERE ---
            # Get the actual target sequence length for the current batch
            target_seq_len = tf.shape(captions_target)[1]
            
            # Slice the model's output to match the target sequence length
            # This ensures both tensors have compatible shapes for loss calculation
            output_sliced = output[:, :target_seq_len, :]
            
            # Get vocab_size dynamically from the sliced output
            current_vocab_size = tf.shape(output_sliced)[2]

            output_reshaped = tf.reshape(output_sliced, [-1, current_vocab_size])
            captions_target_reshaped = tf.reshape(captions_target, [-1])
            # --- FIX ENDS HERE ---
            
            per_token_loss = criterion(captions_target_reshaped, output_reshaped)
            
            mask = tf.cast(tf.not_equal(captions_target_reshaped, dataset.vocab.stoi["<PAD>"]), dtype=tf.float32)
            
            masked_loss = per_token_loss * mask
            
            total_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return total_loss

    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        print_examples(model, dataset, pil_to_tensor_transform)
        
        if save_model:
            save_checkpoint(manager)
            
        for idx, (imgs, captions) in enumerate(train_loader):
            if tf.shape(captions)[1] < 2: 
                print(f"Skipping batch {idx} due to short captions (length < 2).")
                continue

            captions_input = captions[:, :-1] 
            captions_target = captions[:, 1:] 
            
            if tf.shape(captions_target)[1] == 0:
                print(f"Skipping batch {idx} due to empty target captions after slicing.")
                continue

            loss = train_step(imgs, captions_input, captions_target)
            
            with writer.as_default():
                scalar("Training loss", loss, step=step)
            step.assign_add(1)
            
            if idx % 100 == 0:
                print(f"Step {step.numpy()}, Batch {idx}, Loss: {loss.numpy():.4f}")

if __name__ == "__main__":
    if not os.path.exists("flickr8k/images"):
        os.makedirs("flickr8k/images")
        placeholder_img_path = "flickr8k/images/placeholder_image.jpg"
        if not os.path.exists(placeholder_img_path):
            Image.new('RGB', (300, 300), color = 'red').save(placeholder_img_path)

    if not os.path.exists("flickr8k/captions.txt"):
        with open("flickr8k/captions.txt", "w") as f:
            f.write("image,caption\n")
            f.write("placeholder_image.jpg,A dog is running in the grass.\n")
            f.write("placeholder_image.jpg,A brown dog is playing.\n")
            f.write("placeholder_image.jpg,The dog jumps high.\n")
            f.write("placeholder_image.jpg,A cute dog.\n")
            f.write("placeholder_image.jpg,Happy dog.\n")

    if not os.path.exists("test_examples"):
        os.makedirs("test_examples")
        Image.new('RGB', (300, 300), color = 'blue').save("test_examples/dog.jpg")
        Image.new('RGB', (300, 300), color = 'green').save("test_examples/boat.jpg")
        Image.new('RGB', (300, 300), color = 'yellow').save("test_examples/child.jpg")

    train()
