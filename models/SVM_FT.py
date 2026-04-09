import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np

# Tokenization function (with stop word removal)
def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(final)

# Vectorization
vectorizer = TfidfVectorizer(max_features=360)

# Function to evaluate the SVM model on the pre-saved folds for 10-fold cross-validation
def evaluate_svm_on_folds(n_folds=10):
    accuracies = []
    recalls = []
    roc_aucs = []
    f1_scores = []
    confusion_matrices = []

    for fold in range(1, n_folds + 1):
        print(f"Evaluating fold {fold}...")

        # Load training and testing data from the pre-saved parquet files
        train_df = pd.read_parquet(f'text_pdf_except_fold{fold}.parquet')
        test_df = pd.read_parquet(f'text_pdf_fold{fold}.parquet')

        # Preprocess text (punctuation and special character removal)
        train_df['text'] = train_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)
        test_df['text'] = test_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)

        # Replace multiple spaces with a single space
        train_df['text'] = train_df['text'].str.replace('\s+', ' ', regex=True).str.strip()
        test_df['text'] = test_df['text'].str.replace('\s+', ' ', regex=True).str.strip()

        # Tokenize
        train_df['text'] = train_df['text'].apply(tokenize)
        test_df['text'] = test_df['text'].apply(tokenize)

        # Vectorize the text data
        X_train = vectorizer.fit_transform(train_df['text'])
        X_test = vectorizer.transform(test_df['text'])

        # Extract labels
        y_train = train_df['label']
        y_test = test_df['label']

        # SVM classifier with regularisation
        model = SVC(C=1.0, probability=True, class_weight='balanced')  # SVM model
        model.fit(X_train, y_train)

        # Make predictions with custom threshold
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_prob >= 0.005).astype(int)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Collect results for each fold
        accuracies.append(accuracy)
        recalls.append(recall)
        roc_aucs.append(roc_auc)
        f1_scores.append(f1)
        confusion_matrices.append(conf_matrix)

        # Print results for this fold
        print(f"Fold {fold} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")
        print(f"  Confusion Matrix:\n{conf_matrix}\n")

    # Average metrics across all folds
    print(f"\nAverage Metrics Across {n_folds} Folds:")
    print(f"  Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"  Mean Recall: {np.mean(recalls):.4f}")
    print(f"  Mean F1 Score: {np.mean(f1_scores):.4f}")
    print(f"  Mean ROC AUC: {np.mean(roc_aucs):.4f}")

    return accuracies, recalls, f1_scores, roc_aucs, confusion_matrices

# Run the evaluation
evaluate_svm_on_folds(n_folds=10)