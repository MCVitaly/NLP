import json
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForTokenClassification

# Uploading a manually marked-up CoNLL file
def load_conll_data(file_path):
    data = []
    tokens, ner_tags = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                token, label = line.strip().split()
                tokens.append(token)
                ner_tags.append(label)
            else:
                if tokens:
                    data.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens, ner_tags = [], []
    return data

train_data = load_conll_data("furniture_data.conll")
dataset = DatasetDict({"train": Dataset.from_pandas(pd.DataFrame(train_data))})

# Setting up labels and initializing the model
unique_labels = set(label for tags in dataset["train"]["ner_tags"] for label in tags)
label_list = sorted(unique_labels)
label_map = {label: idx for idx, label in enumerate(label_list)}
num_labels = len(label_list)

model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

# tokenizing and aligning labels
def tokenize_and_align_labels(batch):
    tokenized_inputs = tokenizer(batch["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(batch["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label_map[label[word_id]])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# Education parameters
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=8,  #the number of examples processed at the same time
    num_train_epochs=3, #how many times will the model go through the entire training dataset
    weight_decay=0.01,  #the weights will decrease slightly at each iteration
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer
)

# Training and saving model
trainer.train()
model.save_pretrained("./model_output")
tokenizer.save_pretrained("./model_output")

# Saving label list
with open("./model_output/label_list.json", "w") as f:
    json.dump(label_list, f)
