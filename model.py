import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np

# The Vocabulary class from load_data.py is used by the main training script.
# The local Vocabulary class definition is removed to avoid confusion and ensure consistency.

class EncoderCNN(Model):
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = InceptionV3(weights='imagenet', include_top=False)
        self.pool = GlobalAveragePooling2D()
        self.fc = Dense(embed_size)
        self.relu = tf.keras.layers.ReLU()
        self.dropout = Dropout(0.5)
        if not self.train_CNN:
            self.inception.trainable = False

    def call(self, images, training=True):
        features = self.inception(images, training=training)
        features = self.pool(features)
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features, training=training)
        return features


class DecoderRNN(Model):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = Embedding(vocab_size, embed_size)
        self.lstm = LSTM(hidden_size, return_sequences=True, return_state=True)
        self.linear = Dense(vocab_size)
        self.dropout = Dropout(0.5)
        self.hidden_size = hidden_size

    def call(self, features, captions, training=True):
        embeddings = self.dropout(self.embed(captions), training=training)
        features_expanded = tf.expand_dims(features, axis=1)
        embeddings_with_features = tf.concat([features_expanded, embeddings], axis=1)
        hiddens, state_h, state_c = self.lstm(embeddings_with_features, training=training)
        outputs = self.linear(hiddens)
        return outputs, state_h, state_c


class CNNtoRNN(Model):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def call(self, images, captions, training=True):
        features = self.encoderCNN(images, training=training)
        outputs, _, _ = self.decoderRNN(features, captions, training=training)
        return outputs
    
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        if tf.rank(image) == 3:
            image = tf.expand_dims(image, axis=0)
            
        with tf.GradientTape(watch_accessed_variables=False):
            features = self.encoderCNN(image, training=False) 
            input_token = tf.constant([[vocabulary.stoi["<SOS>"]]], dtype=tf.int32)
            
            lstm_state_h = tf.zeros((1, self.decoderRNN.hidden_size), dtype=tf.float32)
            lstm_state_c = tf.zeros((1, self.decoderRNN.hidden_size), dtype=tf.float32)
            states = [lstm_state_h, lstm_state_c]
            
            for i in range(max_length):
                embeddings = self.decoderRNN.embed(input_token)
                
                if i == 0:
                    features_expanded = tf.expand_dims(features, axis=1)
                    lstm_input = tf.concat([features_expanded, embeddings], axis=1)
                else:
                    lstm_input = embeddings
                
                hiddens, lstm_state_h, lstm_state_c = self.decoderRNN.lstm(lstm_input, initial_state=states, training=False)
                states = [lstm_state_h, lstm_state_c]
                
                output_for_linear = hiddens[:, -1, :] 
                
                output = self.decoderRNN.linear(output_for_linear)
                predicted_id = tf.argmax(output, axis=1).numpy()[0]
                
                # FIX: Use vocabulary.itos directly for word lookup
                word = vocabulary.itos[predicted_id]
                result_caption.append(word)
                
                # FIX: Use vocabulary.itos directly for EOS check
                if word == "<EOS>":
                    break
                
                input_token = tf.constant([[predicted_id]], dtype=tf.int32)
                    
        return result_caption


if __name__ == "__main__":
    embed_size = 256
    hidden_size = 512
    num_layers = 1

    # In this __main__ block, we are still using the simple Vocabulary for demonstration.
    # In the actual training loop (train.py), the Vocabulary from load_data.py is used.
    class LocalVocabulary:
        def __init__(self):
            self.stoi = {
                "<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3,
                "a": 4, "cat": 5, "dog": 6, "is": 7, "running": 8,
                "playing": 9, "in": 10, "the": 11, "park": 12, "house": 13
            }
            self.itos = {v: k for k, v in self.stoi.items()}
            self.vocab_size = len(self.stoi)

        def __len__(self):
            return self.vocab_size

        def word_to_idx(self, word):
            return self.stoi.get(word, self.stoi["<UNK>"])

        def idx_to_word(self, idx): # This method is now only relevant for this local test Vocabulary
            return self.itos.get(idx, "<UNK>")

    vocab = LocalVocabulary() # Use the local Vocabulary for testing
    vocab_size = len(vocab)

    sample_image = tf.random.uniform(shape=(299, 299, 3), minval=0, maxval=255, dtype=tf.float32)
    sample_image_processed = tf.keras.applications.inception_v3.preprocess_input(sample_image)
    image_batch = tf.expand_dims(sample_image_processed, axis=0)

    sample_captions = tf.constant([
        [vocab.word_to_idx("<SOS>"), vocab.word_to_idx("cat"), vocab.word_to_idx("is"), vocab.word_to_idx("running"), vocab.word_to_idx("<EOS>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>")],
        [vocab.word_to_idx("<SOS>"), vocab.word_to_idx("dog"), vocab.word_to_idx("playing"), vocab.word_to_idx("in"), vocab.word_to_idx("the"), vocab.word_to_idx("park"), vocab.word_to_idx("<EOS>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>"), vocab.word_to_idx("<PAD>")]
    ], dtype=tf.int32)
    
    print("--- Initializing Model ---")
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)

    print("\n--- Testing Forward Pass (Training Mode) ---")
    _ = model(image_batch, sample_captions, training=True) 
    model.summary()

    outputs = model(image_batch, sample_captions, training=True)
    print(f"Output shape from CNNtoRNN (logits): {outputs.shape}")

    print("\n--- Testing Caption Generation (Inference Mode) ---")
    generated_caption = model.caption_image(sample_image, vocab)
    print(f"Generated Caption: {' '.join(generated_caption)}")

    print("\n--- Testing Caption Generation with another random image ---")
    another_sample_image = tf.random.uniform(shape=(299, 299, 3), minval=0, maxval=255, dtype=tf.float32)
    another_generated_caption = model.caption_image(another_sample_image, vocab)
    print(f"Another Generated Caption: {' '.join(another_generated_caption)}")

    print("\nModel setup complete. You can now train this model with your actual dataset.")
