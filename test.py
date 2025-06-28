import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(test_csv, tokenizer_path, model_path, max_len=150):
    # Load test data
    test_df = pd.read_csv(test_csv)
    # Load tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    # Prepare test data
    X_test = tokenizer.texts_to_sequences(test_df['query'] + ' ' + test_df['image descriptions'])
    X_test = pad_sequences(X_test, maxlen=max_len, padding='post', truncating='post')
    y_test = test_df['Toxic Category Encoded'].values
    # Load model
    model = tf.keras.models.load_model(model_path)
    # Predict
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    # Evaluation
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    # Return DataFrame with predictions
    test_df['Predicted Label'] = y_pred
    return test_df

if __name__ == "__main__":
    test_csv = r"data/test.csv"
    tokenizer_path = r"data/tokenizer.pkl"
    model_path = r"models/toxic_classifier.keras"
    df = evaluate_model(test_csv, tokenizer_path, model_path)
    print(df[['query', 'image descriptions', 'Toxic Category', 'Predicted Label']])