# bash: 
# pip install transformers 
# pip install torch 
# pip install pandas 

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

# **1. Load Pretrained BERT Model and Tokenizer**
bert_model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'  # multilingual bert for chinese user input 
# important 
model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=3)  # num_labels can be adjusted based on your use case

# initialize tokenizer 
tokenizer = BertTokenizer.from_pretrained(bert_model_name)


# **2. Tokenize User Input**
def tokenize_input(text):
    encoded_text = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return encoded_text

def bert_classifier(text): 
    text_tokens = tokenize_input(text)
    print(text_tokens)





# <kaggle ref>  

# from transformers import AutoTokenizer
# # Load Distilbert Tokenizer 
# model_ckpt = "distilbert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

# # Encode Our Example Text
# encoded_text = tokenizer("The movie was not good")
# tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
# print(encoded_text,len(encoded_text.input_ids))
# print(tokens)


# **3. Model Inference**
def classify_text(text):
    inputs = tokenize_input(text)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = softmax(logits, dim=-1)
    return probs

# **4. Category Mapping**
# This is where you'd map the model's output to meaningful categories.
# For demonstration, let's assume we have 3 categories:
categories = ["Brand", "Electronics", "Smartphone"]

def get_category(probs):
    category_index = torch.argmax(probs, dim=-1).item()
    return categories[category_index]

# **5. Result Interpretation**
def interpret_result(text):
    probs = classify_text(text)
    category = get_category(probs)
    return category

# **Example Usage**
user_input = "苹果手机"
category = interpret_result(user_input)
print(f"The input '{user_input}' is categorized as: {category}")