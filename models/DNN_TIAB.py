# Before running this code: install pyarrow and fastparquet

import pandas as pd
import nltk
nltk.download('stopwords')      # If running the code for the first time on a machine
nltk.download('punkt_tab')      # If running the code for the first time on a machine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, f1_score, roc_curve
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import matplotlib.pyplot as plt

# Tokenization function (with stop word removal)
def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(final)

# Vectorization
vectorizer = TfidfVectorizer(max_features=1400)

# Function to create a neural network model
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer=Adam(learning_rate=0.00004), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to evaluate the Deep neural network (DNN) on the pre-saved folds for 10-fold cross-validation
# Training and testing data (for title and abstract) for 10 folds has been saved as 20 different parquet files using a separate code
# Example: Training data file for the 1st fold is titled 'text_csv_except_fold1.parquet', corresponding test data file is titled 'text_csv_fold1.parquet'

def evaluate_dnn_on_folds(n_folds=10):
    accuracies = []
    recalls = []
    roc_aucs = []
    f1_scores = []
    confusion_matrices = []
    all_fprs = []
    all_tprs = []

    for fold in range(1, n_folds + 1):
        print(f"Evaluating fold {fold}...")

        # Load training and testing data from the pre-saved parquet files for 10-fold cross validation
        train_df = pd.read_parquet(f'text_csv_except_fold{fold}.parquet')
        test_df = pd.read_parquet(f'text_csv_fold{fold}.parquet')

        # Preprocess text (punctuation and special character removal)
        train_df['text'] = train_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)
        test_df['text'] = test_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)

        # Tokenize
        train_df['text'] = train_df['text'].apply(tokenize)
        test_df['text'] = test_df['text'].apply(tokenize)

        # Vectorize the text data
        X_train = vectorizer.fit_transform(train_df['text']).toarray()  # Convert to dense matrix for NN
        X_test = vectorizer.transform(test_df['text']).toarray()

        # Extract labels
        y_train = train_df['Label']
        y_test = test_df['Label']

        # Create and train the deep neural network model
        model = create_nn_model(X_train.shape[1])
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

        # Make predictions with custom threshold
        y_pred_prob = model.predict(X_test).flatten()
        y_pred = (y_pred_prob >= 0.00006).astype(int)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        # Collect results for each fold
        accuracies.append(accuracy)
        recalls.append(recall)
        roc_aucs.append(roc_auc)
        f1_scores.append(f1)
        confusion_matrices.append(conf_matrix)
        all_fprs.append(fpr)
        all_tprs.append(tpr)

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

    # Plot ROC curves for each fold
    plt.figure(figsize=(10, 6))
    for i in range(len(all_fprs)):
        plt.plot(all_fprs[i], all_tprs[i], label=f'ROC curve (fold={i+1})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {n_folds} Folds')
    plt.legend(loc="lower right")
    plt.show()

    return accuracies, recalls, f1_scores, roc_aucs, confusion_matrices

# Run the evaluation
evaluate_dnn_on_folds(n_folds=10)