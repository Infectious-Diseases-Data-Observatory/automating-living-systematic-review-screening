import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input, regularizers

# Tokenization function (with stop word removal)
def tokenize(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    final = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(final)

# Vectorization
vectorizer = TfidfVectorizer(max_features=6600)

# Function to create a neural network model on the pre-saved folds for 10-fold cross-validation
# Training and testing data (full text of articles) for 10 folds has been saved as 20 different parquet files using a separate code
# Example: Training data file for the 1st fold is titled 'text_pdf_except_fold1.parquet', corresponding test data file is titled 'text_pdf_fold1.parquet'

def create_nn_model(input_dim):
    model = Sequential()
    model.add(Input(shape=(input_dim,)))
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.05))  # Dropout layer to prevent overfitting
    model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.05))
    model.add(Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer=Adam(learning_rate=0.00009), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to evaluate the DNN model on the pre-saved folds
def evaluate_dnn_on_folds(n_folds=10):
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

        # Neural Network classifier
        model = create_nn_model(X_train.shape[1])
        model.fit(X_train.toarray(), y_train, epochs=40, batch_size=32, verbose=0)

        # Make predictions with custom threshold
        y_pred_prob = model.predict(X_test.toarray()).flatten()
        y_pred = (y_pred_prob >= 0.02).astype(int)

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
evaluate_dnn_on_folds(n_folds=10)