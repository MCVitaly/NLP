import csv
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

model_path = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()  # switching to inference mode

# Loading a list of labels
with open(f"{model_path}/label_list.json", "r") as f:
    label_list = json.load(f)


def label_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():  # disabling the calculation of gradients
        outputs = model(**tokens).logits    #calls a model with tokenized input data
    predictions = outputs.argmax(dim=-1).squeeze().tolist() #extracting predicted labels from logisticians obtained from the model
    labels = [label_list[i] for i in predictions if i != -100]

    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(), skip_special_tokens=True)
    cleaned_words = []
    cleaned_labels = []

    current_word = ""
    current_label = None

    for word, label in zip(words, labels):
        if word.startswith("##"):
            current_word += word[2:]
        else:
            if current_word:
                cleaned_words.append(current_word)
                cleaned_labels.append(current_label)
            current_word = word
            current_label = label

    if current_word:
        cleaned_words.append(current_word)
        cleaned_labels.append(current_label)

    return list(zip(cleaned_words, cleaned_labels))


def auto_annotate(limit_rows=None):
    with open("furniture_data.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for i, row in enumerate(reader) if limit_rows is None or i < limit_rows]

    with open("furniture_data.conll", "w", encoding="utf-8") as f:
        for row in rows:
            text = " ".join(row)
            labeled_data = label_text(text)
            for word, label in labeled_data:
                f.write(f"{word} {label}\n")

    with open("furniture_data.csv", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        remaining_rows = [row for i, row in enumerate(reader) if limit_rows is None or i >= limit_rows]

    with open("furniture_data.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(remaining_rows)


if __name__ == "__main__":
    # True to process only first 1000 lines, False to process fill file
    process_first_1000 = True

    limit_rows = 1000 if process_first_1000 else None
    auto_annotate(limit_rows=limit_rows)
