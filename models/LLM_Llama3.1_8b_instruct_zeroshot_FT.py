import transformers
import torch
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix

# Load data
data = pd.read_parquet('text.parquet')

# Print basic data info
print("Number of rows in the dataset:", data.shape[0])
print("\nColumn names: ", data.columns)

df = data.copy()

# LLM used
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# A new access token will have to be generated on huggingface.co and used, if this one has expired. All user permissions under Settings > Access tokens > repositories, need to be enabled
access_token = "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"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token=access_token,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# List to store predictions
predicted_label = []

# Either Prompt A or Prompt B can be used
# Prompt A
prompt = "A systematic review of studies on treatment of Malaria needs to be done. Only those research articles which present a primary antimalarial drug treatment efficacy clinical trial should be included. Articles which are literature reviews, case reports, editorials, commentaries, correspondence, letters to the editor, opinion pieces, corrections of previous research articles, or which do not present a primary study on the efficacy of an antimalarial drug must be excluded. It is important to understand that an article may talk about the treatment of Malaria, but only if it presents a primary clinical trial of an antimalarial drug, it should be included. Articles on secondary studies must not be included. The abstract of a research article is provided, which you need to read and make a decision. Answer 'Include' or 'Exclude'. Here is the abstract:"

'''
# Prompt B : For discouraging the model to easily exclude articles
prompt = "A systematic review of studies on malaria treatment needs to be conducted. Research articles presenting primary clinical trials on the efficacy of antimalarial drugs must be included. Articles that are literature reviews, case reports, editorials, commentaries, letters, opinion pieces, corrections, or that do not present a primary study on antimalarial drug efficacy should be excluded. Do not easily exclude an article. Exclude it only if there is strong evidence in support of its exclusion. The abstract of a research article is provided. Read it and make a decision. Answer 'Include' or 'Exclude' based on the given abstract of the paper. Here is the abstract:"
'''

c = 0

for abstract in df['text']:
    promptf = prompt + abstract[:3000]      # Token limit of 3000 can be increased if computation resources allow

    messages = [
        {"role": "user", "content": promptf},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=1,
    )

    if "Include" in outputs[0]["generated_text"][-1]["content"]:
        llm_class0 = 1
    else:
        llm_class0 = 0

    predicted_label.append(llm_class0)

    c = c + 1

    if c%100==0:
        hundred = c/100
        print("Hundreds complete:", hundred)    # Prints number of hundred predictions completed

# Add a column to the dataframe df for predicted labels
df['predicted_label'] = predicted_label

# Calculate performance metrics
accuracy = accuracy_score(df['label'], df['predicted_label'])
recall = recall_score(df['label'], df['predicted_label'])
precision = precision_score(df['label'], df['predicted_label'])
f1 = f1_score(df['label'], df['predicted_label'])
roc_auc = roc_auc_score(df['label'], df['predicted_label'])
conf_matrix = confusion_matrix(df['label'], df['predicted_label'])
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Exclude', 'Actual Include'], columns=['Predicted Exclude', 'Predicted Include'])

# Print performance metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix_df)