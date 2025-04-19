!pip install transformers
from transformers import pipeline

translator = pipeline("translation", model = "Helsinki-NLP/opus-mt-en-hi")

user_input = input("Enter a sentence in English: ")
result = translator(user_input)

print("Translated text:", result[0]['translation_text'])
