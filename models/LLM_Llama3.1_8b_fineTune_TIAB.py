# This code was written to run on google collab

# We need unsloth and xformers packages but latest torch will not work so remove latest torch and install...
# older version before installing unsloth and xformers

# Chunk 1
!pip uninstall torch
!pip uninstall pyarrow

!pip install torch==2.3.0
!pip install "triton"

!pip install pyarrow==14.0.1
!pip install bitsandbytes
!pip install trl
!pip install peft

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27"


# Chunk 2
from unsloth import FastLanguageModel
import torch

max_seq_length=2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit
)

model = FastLanguageModel.get_peft_model (
          model,
          target_modules=["q_proj", "k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj",],
          lora_alpha = 16,
          lora_dropout = 0,
          bias = "none",
          use_gradient_checkpointing = "unsloth",
          random_state = 3407,
          use_rslora = False,
          loftq_config = None
                   )

# Either Prompt A or Prompt B can be used
# Prompt A
# alpaca_prompt_ip = """A systematic review of studies on the treatment of malaria needs to be conducted. Only research articles presenting primary clinical trials on the efficacy of antimalarial drugs should be included. Articles that are literature reviews, case reports, editorials, commentaries, correspondence, letters to the editor, opinion pieces, corrections of previous research articles, or that do not present a primary study on the efficacy of an antimalarial drug must be excluded. It is crucial to understand that while an article may discuss the treatment of malaria, it should only be included if it presents a primary clinical trial of an antimalarial drug. Articles on secondary studies must not be included. The abstract of a research article is provided, and you need to read it and make a decision. Answer 'Include' or 'Exclude' based on the given abstract of the paper.

# Prompt B
alpaca_prompt_ip = """A systematic review of studies on malaria treatment needs to be conducted. Research articles presenting primary clinical trials on the efficacy of antimalarial drugs must be included. Articles that are literature reviews, case reports, editorials, commentaries, letters, opinion pieces, corrections, or that do not present a primary study on antimalarial drug efficacy should be excluded. Do not easily exclude an article. Exclude it only if there is strong evidence in support of its exclusion. The abstract of a research article is provided, read it and make a decision. Answer 'Include' or 'Exclude' based on the given abstract of the paper.


###Abstract:
{}

###Input:
{}

###Response:
{}"""


EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
  abstracts = examples["Abstract"]
  inputs = examples["Input"]
  outputs = examples["Response"]

  texts = []
  for abstract, input, output in zip(abstracts, inputs, outputs):
    text = alpaca_prompt_ip.format(abstract, input, output) + EOS_TOKEN
    texts.append(text)
  return {"text": texts}

from datasets import load_dataset

#This implementation is for one of the five folds of title and abstract data
# Training dataset file balanced_dataset_2345.csv consists of folds 2 through 5
# Corresponding Testing dataset file is balanced_dataset_1_num.csv (fold 1)

dataset = load_dataset("csv", data_files="balanced_dataset_2345.csv", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)


#Chunk 3
from trl import SFTTrainer

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer ( model = model,
                      tokenizer = tokenizer,
                      train_dataset = dataset,
                      dataset_text_field="text",
                      max_seq_length = max_seq_length,
                      dataset_num_proc = 2,
                      packing = False,
                      args = TrainingArguments(
                          per_device_train_batch_size=16,
                          #gradient_accumulation_steps=4 ,
                          warmup_steps = 30,
                          max_steps  = 700,
                          learning_rate = 2e-4,
                          fp16= not is_bfloat16_supported(),
                          bf16=is_bfloat16_supported(),
                          logging_steps=1,
                          optim="adamw_8bit",
                          weight_decay=0.01,
                          lr_scheduler_type="linear",
                          seed=27,
                          output_dir = "outputs"
                      ),
                      )


trainer_stats = trainer.train()


FastLanguageModel.for_inference(model)


import re

def extract_after_pattern(text, pattern):
    # Compile the regular expression to find the pattern
    regex = re.compile(rf'{pattern}(.*)', re.DOTALL)

    # Search for the pattern in the text
    match = regex.search(text)

    if match:
        # Extract the part after the pattern
        result = match.group(1).strip()

        # Remove the "<|end_of_text|>" marker if present
        result = re.sub(r'<\|end_of_text\|>', '', result)

        # Remove special characters (excluding alphanumeric and basic punctuation)
        result = re.sub(r'[^\w\s,.?!]', '', result)

        return result.strip()
    else:
        return None


# Chunk 4
vald = load_dataset("csv", data_files="balanced_dataset_1_num.csv", split="train")

generated_text_llm = []
predicted_label_llm = []

for abst in vald['Abstract']:
  inputs = tokenizer( [alpaca_prompt_ip.format(abst,"","")],return_tensors="pt").to("cuda")
  outputs = model.generate(**inputs, max_new_tokens=16, use_cache=True)
  text = tokenizer.batch_decode(outputs)[0]
  generated_text0 = extract_after_pattern(text, "###Response:")

  if "Include" in generated_text0:
      llm_class0 = 1
  else:
      llm_class0 = 0
  generated_text_llm.append(generated_text0)
  predicted_label_llm.append(llm_class0)


# Chunk 5
df1 = vald.to_pandas()
df1['predicted_label_llm'] = predicted_label_llm
df1['generated_text_llm'] = generated_text_llm


# Chunk 6
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd

# Calculate metrics
accuracy = accuracy_score(df1['Response'], df1['predicted_label_llm'])
recall = recall_score(df1['Response'], df1['predicted_label_llm'])
precision = precision_score(df1['Response'], df1['predicted_label_llm'])
f1 = f1_score(df1['Response'], df1['predicted_label_llm'])
roc_auc = roc_auc_score(df1['Response'], df1['predicted_label_llm'])
conf_matrix = confusion_matrix(df1['Response'], df1['predicted_label_llm'])

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Exclude', 'Actual Include'], columns=['Predicted Exclude', 'Predicted Include'])
print("Confusion Matrix:")
print(conf_matrix_df)


