import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Set paths
image_dir = 'path_to_images/'
annotations_file = 'path_to_annotations/annotations.txt'

# Preprocess images (use InceptionV3 for feature extraction)
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(299, 299))  # Resize to InceptionV3 input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

# Load InceptionV3 model pre-trained on ImageNet for feature extraction
def extract_image_features(image_dir):
    model_incep = InceptionV3(weights='imagenet')
    new_input = model_incep.input
    hidden_layer = model_incep.layers[-2].output
    feature_extractor = Model(inputs=new_input, outputs=hidden_layer)

    features = {}
    for img_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, img_name)
        img = preprocess_image(image_path)
        feature = feature_extractor.predict(img, verbose=0)
        features[img_name] = feature
    return features

# Preprocess captions and tokenize
def preprocess_captions(annotations_file):
    captions = {}
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        img_id, caption = parts[0], parts[1]
        if img_id not in captions:
            captions[img_id] = []
        captions[img_id].append(caption)
    return captions

# Tokenize captions
def tokenize_captions(captions):
    tokenizer = Tokenizer()
    all_captions = []
    for key, caps in captions.items():
        for cap in caps:
            all_captions.append(cap)
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer, vocab_size

# Prepare sequences for training
def create_sequences(captions, tokenizer, vocab_size, max_sequence_length=34):
    sequences = []
    image_ids = []
    for key, caps in captions.items():
        img_id = key.split('.')[0]
        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_sequence_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                sequences.append([features[img_id + '.jpg'], in_seq, out_seq])
                image_ids.append(img_id)
    return np.array(sequences), image_ids

# Load image features and captions
features = extract_image_features(image_dir)
captions = preprocess_captions(annotations_file)

# Tokenize and prepare sequences
tokenizer, vocab_size = tokenize_captions(captions)
sequences, image_ids = create_sequences(captions, tokenizer, vocab_size)

# Build a simple LSTM model
def build_captioning_model(vocab_size, max_sequence_length):
    # Image feature input
    image_input = tf.keras.Input(shape=(2048,))
    image_layer = Dense(256, activation='relu')(image_input)
    
    # Caption input
    caption_input = tf.keras.Input(shape=(max_sequence_length,))
    caption_embedding = Embedding(vocab_size, 256)(caption_input)
    caption_lstm = LSTM(256)(caption_embedding)
    
    # Combine image features and caption
    combined = tf.keras.layers.add([image_layer, caption_lstm])
    combined = Dropout(0.5)(combined)
    output = Dense(vocab_size, activation='softmax')(combined)
    
    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Train the model
model = build_captioning_model(vocab_size, max_sequence_length=34)
model.fit([sequences[:, 0], sequences[:, 1]], sequences[:, 2], epochs=20, batch_size=64)

# Function to generate caption for a new image
def generate_caption(model, image_path, tokenizer, max_sequence_length=34):
    image_feature = extract_image_features(image_path)
    image_feature = image_feature[image_path.split('/')[-1]]  # Get feature of the specific image

    in_seq = [tokenizer.word_index['startseq']]  # Start sequence token
    for i in range(max_sequence_length):
        seq = pad_sequences([in_seq], maxlen=max_sequence_length)
        pred = model.predict([image_feature, seq], verbose=0)
        pred_idx = np.argmax(pred)
        pred_word = tokenizer.index_word[pred_idx]

        if pred_word == 'endseq':
            break

        in_seq.append(pred_idx)

    caption = ' '.join([tokenizer.index_word[idx] for idx in in_seq[1:]])
    return caption

# Example usage
image_path = 'path_to_flickr_image.jpg'
caption = generate_caption(model, image_path, tokenizer)
print("Generated Caption:", caption)
