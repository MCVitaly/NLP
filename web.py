from flask import Flask, request, jsonify, render_template_string
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)


model_path = "./model_output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()


def extract_product_names(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**tokens).logits
    predictions = outputs.argmax(dim=-1).squeeze().tolist()

    # Extracting predicted product names
    product_names = []
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(), skip_special_tokens=True)
    for word, prediction in zip(words, predictions):
        if prediction == 1:
            cleaned_word = word.replace("##", "")
            product_names.append(cleaned_word)

    return " ".join(product_names)

# The route for the main page with the form
@app.route("/")
def index():
    return render_template_string("""
        <form action="/extract" method="post">
            <label for="url">Enter URL:</label>
            <input type="text" name="url" id="url" required>
            <button type="submit">Extract products</button>
        </form>
    """)

# The route for URL processing and product extraction
@app.route("/extract", methods=["POST"])
def extract():
    url = request.form.get("url")
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
        product_names = extract_product_names(text)
        return jsonify({"products": product_names})
    except requests.exceptions.RequestException:
        return jsonify({"error": "The URL content could not be loaded"}), 500

if __name__ == "__main__":
    app.run(debug=True)
