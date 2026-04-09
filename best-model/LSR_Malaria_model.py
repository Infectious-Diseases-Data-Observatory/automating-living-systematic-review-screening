# Deep Neural Network code for screening research articles for the following Living Systematic review (LSR) update:
# The WWARN Clinical Trials Library: A systematically constructed database of clinical efficacy trials of human-infecting Plasmodium Public registration Updates
# (https://osf.io/fpx9t)
# Important: Though the training data for the model comprises only Title and Abstract and not full text,
# but the prediction by the model is for finally including or excluding the article for the LSR

# Before running this code: install pyarrow (e.g. pip install pyarrow)

import pandas as pd
import ftfy
import re
import nltk
nltk.download('stopwords')      # If running the code for the first time on a machine
nltk.download('punkt_tab')      # If running the code for the first time on a machine
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
from sklearn.model_selection import train_test_split

# Training and testing data is in the file 'Title_and_Abstract.csv'. This file was created by concatenating the three 'included', 'excluded', and 'irrelevant'...
# ...csv files downloaded from Covidence. After concatenation, about 36 rows with missing abstracts were manually filled with the respective abstracts.
# ...this was followed by dropping any row that was left with missing Title or Abstract.
# Read the CSV file with latin1 encoding as most other encodings did not work
data = pd.read_csv('Title_and_Abstract.csv', encoding='latin1')

# Function to fix text encoding issues using ftfy
def fix_text_encoding(text):
    try:
        # First, fix common encoding issues with ftfy
        text = ftfy.fix_text(text)
        return text
    except Exception as e:
        return text

# Custom replacements for known problematic sequences
def custom_replacements(text):
    replacements = {
        'Â€Â“': '–',
        'Â€Â“': '–',
        'aÂ€Â“': '–',
        'a€"': '–',
        'Â': '',
        'â€™': '’',
        'â€“': '–',
        'â€': '"',
        'â€¢': '•',
        'â€ ': '-',
        'â€ ': '"',
        'â€”': '—',
        'â€': '€',
        'â€™': "'",
        'â€œ': '"',
        'â€˜': "'",
        'â€¦': '...',
        'â‚¬': '€',
        'â€"': '–'
    }
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

# Combine ftfy and custom replacements
def fix_and_replace(text):
    return custom_replacements(fix_text_encoding(text))

# Fix encoding issues by applying the above function to relevant columns
data['Title'] = data['Title'].apply(fix_and_replace)
data['Abstract'] = data['Abstract'].apply(fix_and_replace)

# Replace multiple spaces with a single space
data['Title'] = data['Title'].str.replace('\s+', ' ', regex=True).str.strip()
data['Abstract'] = data['Abstract'].str.replace('\s+', ' ', regex=True).str.strip()

# Combine 'Title' and 'Abstract' into a single feature for simplicity
data['text'] = data['Title'] + ' ' + data['Abstract']

y = data['Label']

# Randomly split data for training and testing while ensuring each split is stratified
train_df, test_df, y_train, y_test = train_test_split(data, y, test_size=0.1, random_state=27, stratify=y)

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

print(f"Training the model...")

# Preprocess text (punctuation and special character removal). Numbers, and certain symbols are retained as these may carry useful information
train_df['text'] = train_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)
test_df['text'] = test_df['text'].str.replace('[\[\]\(\)]', ' ', regex=True).str.replace('[^\w\s\d%/<>-]', '', regex=True)

# Tokenize
train_df['text'] = train_df['text'].apply(tokenize)
test_df['text'] = test_df['text'].apply(tokenize)

# Vectorize the text data
X_train = vectorizer.fit_transform(train_df['text']).toarray()  # Convert to dense matrix for Deep Neural Network
X_test = vectorizer.transform(test_df['text']).toarray()

# Create and train the neural network model
model = create_nn_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

print(f"Testing the model...")

# Make predictions for validation
y_pred_prob = model.predict(X_test).flatten()

# (WHILE VALIDATING THE MODEL AGAINST LSR UPDATE,
# IF IT FAILS TO INCLUDE ANY RESEARCH ARTICLE THAT WAS MANUALLY INCLUDED, CHANGE THIS CUT-OFF TO 0.00005 AND CHECK AGAIN
# IF CHANGING THIS CUT-OFF HERE, ALSO CHANGE IT IN THE FINAL CHUNK OF THIS CODE USED TO MAKE PREDICTION ON LSR UPDATE)
y_pred = (y_pred_prob >= 0.00006).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results for test data
print(f"Test results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1 Score: {f1:.4f}")
print(f"  ROC AUC: {roc_auc:.4f}")
print(f"  Confusion Matrix:\n{conf_matrix}\n")

# # # # # # # #

# PREDICTION ON LSR UPDATE
print(f"Inference on new data...")

# Read the data (LSR_update_data.csv) for inference
data_for_prediction = pd.read_csv('LSR_update_data.csv', encoding='latin1')

# Strip leading and trailing whitespaces from 'Title' and 'Abstract'
data_for_prediction['Title'] = data_for_prediction['Title'].str.strip()
data_for_prediction['Abstract'] = data_for_prediction['Abstract'].str.strip()

# Initialize the 'Prediction' column with default value 'Manual check needed'
# If any row in the uploaded csv file has a missing Title or Abstract, model will assign 'Manual check needed' to the row under the 'Prediction' column
# This would be done as without Title or Abstract, model doesn't have reliable information to make prediction, hence this articles needs manual screening
data_for_prediction['Prediction'] = 'Manual check needed'

# Iterate over rows and process those with valid 'Title' and 'Abstract'
for index, row in data_for_prediction.iterrows():
    title = row['Title']
    abstract = row['Abstract']

    # Check if 'Title' and 'Abstract' are not empty
    if pd.notna(title) and pd.notna(abstract) and title != '' and abstract != '':
        # Combine 'Title' and 'Abstract' into a single feature for model input
        combined_text = title + ' ' + abstract

        # Preprocess and tokenize the text
        combined_text_cleaned = fix_and_replace(combined_text)

        # Replace special characters using re.sub()
        combined_text_cleaned = re.sub(r'[\[\]\(\)]', ' ', combined_text_cleaned)  # Replace brackets and parentheses with space
        combined_text_cleaned = re.sub(r'[^\w\s\d%/<>-]', '', combined_text_cleaned)  # Remove non-alphanumeric characters except those specified

        combined_text_cleaned = tokenize(combined_text_cleaned)

        # Vectorize the combined text
        X_holdout = vectorizer.transform([combined_text_cleaned]).toarray()

        # Make prediction using the pre-trained model
        y_holdout_pred_prob = model.predict(X_holdout).flatten()
        y_holdout_pred = (y_holdout_pred_prob >= 0.00006).astype(int)

        # A new column 'Prediction' will be added to LSR_update_data.csv file.
        # Model will assign 'Include' to a row under this column if the research article corresponding to the row should be included in the LSR
        # else, model will assign 'Exclude'
        data_for_prediction.at[index, 'Prediction'] = 'Include' if y_holdout_pred == 1 else 'Exclude'

# Save the LSR update data with predictions to a CSV file
data_for_prediction.to_csv('LSR_update_data_with_predictions.csv', index=False)

print("Data for inference saved to 'LSR_update_data_with_predictions.csv'.")
