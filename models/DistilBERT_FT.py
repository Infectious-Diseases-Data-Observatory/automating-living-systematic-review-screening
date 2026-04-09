from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch import nn
import numpy as np

# Load the tokenizer and the model
model_id = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Function to tokenize the dataset
class ModelDataset(torch.utils.data.Dataset):
    def __init__(self, notes, labels, tokenizer):
        self.notes = notes
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.notes)

    def __getitem__(self, item):
        note = self.notes[item]
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
            note,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'label': torch.tensor(int(label))}

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=9e-6,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=6,
    weight_decay=0.035,
    logging_dir='./logs',
    logging_steps=100,  # Log less frequently (every 100 steps)
    max_grad_norm=6.0,
    fp16=True,
    no_cuda=False,
    disable_tqdm=True  # Disable the progress bars and intermediate logging
)

# Custom Trainer with class weights to balance minority and majority class weights
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

# Metrics function
def compute_metrics_with_roc_auc(predictions):
    logits = predictions.predictions
    labels = predictions.label_ids
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    positive_probs = probabilities[:, 1]
    preds = (positive_probs >= 0.02).astype(int)

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    roc_auc = roc_auc_score(labels, positive_probs)
    conf_matrix = confusion_matrix(labels, preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'eval_conf_matrix': conf_matrix.tolist()
    }

# Initialize a variable to accumulate the confusion matrices
combined_conf_matrix = np.zeros((2, 2))

# Training and testing data (full text of articles) for 10 folds has been saved as 20 different parquet files using a separate code
# Example: Training data file for the 1st fold is titled 'text_pdf_except_fold1.parquet', corresponding test data file is titled 'text_pdf_fold1.parquet'
# Loop through 10 folds, reading the parquet files for each fold

results = []

for fold in range(1, 11):
    print(f"Training fold {fold}...")

    # Load training and validation (test) datasets from the parquet files
    train_df = pd.read_parquet(f'text_pdf_except_fold{fold}.parquet')
    val_df = pd.read_parquet(f'text_pdf_fold{fold}.parquet')

    # Calculate class weights
    class_counts = train_df['label'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    class_weights = torch.tensor([total_samples / class_count for class_count in class_counts], dtype=torch.float)

    # Create datasets in a form which can be used by the model
    train_dataset = ModelDataset(train_df['text'].values, train_df['label'].values, tokenizer=tokenizer)
    val_dataset = ModelDataset(val_df['text'].values, val_df['label'].values, tokenizer=tokenizer)

    # Load the model for each fold
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
    )

    # Trainer for this fold
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        class_weights=class_weights,
        compute_metrics=compute_metrics_with_roc_auc
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Fold {fold} Validation Results: {eval_results}")

    # Append fold results to the results list
    results.append(eval_results)

    # Add confusion matrix from this fold to the cumulative matrix
    combined_conf_matrix += np.array(eval_results['eval_conf_matrix'])


# Calculate average performance across all 10 folds
avg_results = {
    'eval_loss': sum([r['eval_loss'] for r in results]) / 10,
    'eval_accuracy': sum([r['eval_accuracy'] for r in results]) / 10,
    'eval_precision': sum([r['eval_precision'] for r in results]) / 10,
    'eval_recall': sum([r['eval_recall'] for r in results]) / 10,
    'eval_f1': sum([r['eval_f1'] for r in results]) / 10,
    'eval_roc_auc': sum([r['eval_roc_auc'] for r in results]) / 10,
}

# Print average performance for 10 folds
print(f"Average Cross-Validation Results: {avg_results}")

conf_matrix_df = pd.DataFrame(combined_conf_matrix, index=['Actual Exclude', 'Actual Include'], columns=['Predicted Exclude', 'Predicted Include'])
print("Combined Confusion Matrix across all folds:")
print(conf_matrix_df)
