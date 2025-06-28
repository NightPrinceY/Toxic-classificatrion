import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def build_model(vocab_size, max_len, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_path = os.path.join(data_dir, 'train.csv')
    eval_path = os.path.join(data_dir, 'eval.csv')
    tokenizer_path = os.path.join(data_dir, 'tokenizer.pkl')

    # Load data
    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    max_words = 10000
    max_len = 150

    # Prepare text and labels
    X_train = tokenizer.texts_to_sequences(train_df['query'] + ' ' + train_df['image descriptions'])
    X_eval = tokenizer.texts_to_sequences(eval_df['query'] + ' ' + eval_df['image descriptions'])
    X_train = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_eval = pad_sequences(X_eval, maxlen=max_len, padding='post', truncating='post')
    y_train = train_df['Toxic Category Encoded'].values
    y_eval = eval_df['Toxic Category Encoded'].values

    num_classes = 9  # There are 9 classes in your dataset

    # Build and train model
    model = build_model(max_words, max_len, num_classes)
    early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_eval, y_eval),
        callbacks=[early_stop]
    )

    # Save model
    model.save(os.path.join(os.path.dirname(__file__), 'models/toxic_classifier.keras'))