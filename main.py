import argparse
import os
import re
import pandas as pd
import joblib
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem import WordNetLemmatizer
import nltk


nltk.download('wordnet')
nltk.download('omw-1.4')


def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text))
    text = text.lower()
    words = text.split()
    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized)


def load_data(file_path, max_rows=None):
    print(f"Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)
    if max_rows:
        df = df.head(max_rows)
    print(f"Loaded {len(df)} rows (limited to max_rows={max_rows})")
    df = df.dropna(subset=["Consumer complaint narrative"])
    return df


def preprocess_data(df):
    print("Preprocessing data...")

    
    df["Product"] = df["Product"].replace({
        "Credit card or prepaid card": "Credit card",
        "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
        "Payday loan, title loan, personal loan, or advance loan": "Payday loan",
        "Money transfer, virtual currency, or money service": "Money transfer",
    })

    
    df["Consumer complaint narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    X = df["Consumer complaint narrative"]
    y = df["Product"]

    
    counts = Counter(y)
    valid_classes = [cls for cls, c in counts.items() if c >= 2]
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    print(f"After removing rare classes, {len(y)} samples remain across {len(valid_classes)} classes.")
    return X, y


def train_model(X_train, y_train, epochs):
    print(f"Training model for {epochs} epochs...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=7000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=epochs * 200, class_weight='balanced', solver='liblinear')
    model.fit(X_train_vec, y_train)

    return model, vectorizer


def evaluate_model(model, vectorizer, X_test, y_test):
    print("Evaluating model...")
    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
    print("\nSample predictions:  ")
    sample_tests = X_test[:5]
    true_labels = y_test[:5]
    sample_preds = model.predict(vectorizer.transform(sample_tests))
    for text, true, pred in zip(sample_tests, true_labels, sample_preds):
        print(f"Text: {text[:75]}... | True: {true} | Predicted: {pred}")
        print("")

    


def save_outputs(model, vectorizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, "model.joblib"))
    joblib.dump(vectorizer, os.path.join(out_dir, "vectorizer.joblib"))
    print(f"Saved model and vectorizer to {out_dir}/")


def main(args):
    df = load_data(args.file_path, args.max_rows)
    X, y = preprocess_data(df)


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, vectorizer = train_model(X_train, y_train, args.epochs)
    evaluate_model(model, vectorizer, X_test, y_test)
    save_outputs(model, vectorizer, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaiburr Task 5 - Complaint Classification (Improved Version)")
    parser.add_argument("--file_path", type=str, required=True, help="Path to complaints.csv file")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_rows", type=int, default=None, help="Maximum number of rows to load")
    args = parser.parse_args()
    main(args)




