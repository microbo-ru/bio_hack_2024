from textwrap import wrap
import os

import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.noise_scheduler import NoiseScheduler
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras


if __name__ == '__main__':
    print("start")
    # keras_cv.models.stable_diffusion.TextEncoder

    df = pd.read_csv('hands/HandInfo.csv')
    print(df)

    def combine_columns(row):
        return str(row['gender']) + ' ' + row['aspectOfHand']

    df['caption'] = df.apply(combine_columns, axis=1)
    df['image_path'] = df['imageName'].apply(lambda t: f'hands/Hands/Hands/{t}')

    data_frame = df[['image_path', 'caption']]
    # data_frame.rename({"imageName": "image_path"}, axis=1, inplace=True)
    # print(data_frame)
    # exit()

    PADDING_TOKEN = 49407
    MAX_PROMPT_LENGTH = 77

    # Load the tokenizer.
    
    tokenizer = keras_cv.models.stable_diffusion.SimpleTokenizer()

    #  Method to tokenize and pad the tokens.
    def process_text(caption):
        tokens = tokenizer.encode(caption)
        tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH - len(tokens))
        return np.array(tokens)


    # Collate the tokenized captions into an array.
    tokenized_texts = np.empty((len(data_frame), MAX_PROMPT_LENGTH))

    all_captions = list(data_frame["caption"].values)
    for i, caption in enumerate(all_captions): 
        tokenized_texts[i] = process_text(caption)

################
    RESOLUTION = 256
    AUTO = tf.data.AUTOTUNE
    POS_IDS = tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

    augmenter = keras.Sequential(
        layers=[
            keras_cv.layers.CenterCrop(RESOLUTION, RESOLUTION),
            keras_cv.layers.RandomFlip(),
            tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1),
        ]
    )
    text_encoder = TextEncoder(MAX_PROMPT_LENGTH)


    def process_image(image_path, tokenized_text):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, 3)
        image = tf.image.resize(image, (RESOLUTION, RESOLUTION))
        return image, tokenized_text


    def apply_augmentation(image_batch, token_batch):
        return augmenter(image_batch), token_batch


    def run_text_encoder(image_batch, token_batch):
        return (
            image_batch,
            token_batch,
            text_encoder([token_batch, POS_IDS], training=False),
        )


    def prepare_dict(image_batch, token_batch, encoded_text_batch):
        return {
            "images": image_batch,
            "tokens": token_batch,
            "encoded_text": encoded_text_batch,
        }


    def prepare_dataset(image_paths, tokenized_texts, batch_size=1):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, tokenized_texts))
        dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.map(process_image, num_parallel_calls=AUTO).batch(batch_size)
        dataset = dataset.map(apply_augmentation, num_parallel_calls=AUTO)
        dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
        dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO)
        return dataset.prefetch(AUTO)
    
    training_dataset = prepare_dataset(
        np.array(data_frame["image_path"]), tokenized_texts, batch_size=4
    )

    # Take a sample batch and investigate.
    sample_batch = next(iter(training_dataset))

    for k in sample_batch:
        print(k, sample_batch[k].shape)

    plt.figure(figsize=(20, 10))
    for i in range(3):
        ax = plt.subplot(1, 4, i + 1)
        plt.imshow((sample_batch["images"][i] + 1) / 2)

        text = tokenizer.decode(sample_batch["tokens"][i].numpy().squeeze())
        text = text.replace("<|startoftext|>", "")
        text = text.replace("<|endoftext|>", "")
        text = "\n".join(wrap(text, 12))
        plt.title(text, fontsize=15)

        plt.axis("off")