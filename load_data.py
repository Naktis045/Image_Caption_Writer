import tensorflow as tf
import numpy as np
import os
import pandas as pd
import collections
from PIL import Image

class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = 4

        for sentence in sentence_list:
            for word in sentence.lower().split():
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = text.lower().split()
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class FlickrDataset(tf.keras.utils.Sequence):
    def __init__(self, root_folder, annotation_file, transform=None, freq_threshold=5):
        self.root_folder = root_folder
        self.df = pd.read_csv(annotation_file)
        self.transform = transform

        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]

        img_path = os.path.join(self.root_folder, img_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption += [self.vocab.stoi["<EOS>"]]

        return img, tf.constant(numericalized_caption, dtype=tf.int64)


def get_loader(root_folder, annotation_file, transform, batch_size=32, shuffle=True):
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)

    def dataset_generator():
        for i in range(len(dataset)):
            yield dataset[i]

    output_signature = (
        tf.TensorSpec(shape=(299, 299, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )

    tf_dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=output_signature
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)

    tf_dataset = tf_dataset.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([299, 299, 3]), tf.TensorShape([None])),
        padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(dataset.vocab.stoi["<PAD>"], dtype=tf.int64))
    )

    tf_dataset = tf_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return tf_dataset, dataset
