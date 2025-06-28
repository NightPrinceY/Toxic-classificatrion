from src.preprocess import preprocess
from src.split_data import split_data
from src.tokenize import create_and_save_tokenizer
from models.model import train_model

def main():
    # Step 1: Clean and preprocess data
    preprocess(
        input_csv='data/cellula-toxic.csv',
        output_csv='data/cleaned.csv'
    )
    # Step 2: Split data
    split_data(
        cleaned_csv='data/cleaned.csv',
        train_csv='data/train.csv',
        eval_csv='data/eval.csv',
        test_csv='data/test.csv'
    )
    # Step 3: Tokenize and save tokenizer
    create_and_save_tokenizer(
        train_csv='data/train.csv',
        tokenizer_path='data/tokenizer.pkl'
    )
    # Step 4: Train model
    train_model(
        train_csv='data/train.csv',
        eval_csv='data/eval.csv',
        tokenizer_path='data/tokenizer.pkl',
        model_path='models/toxic_classifier.keras'
    )

if __name__ == "__main__":
    main()