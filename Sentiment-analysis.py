!pip install transformers
from transformers import pipeline
Classifier = pipeline("sentiment-analysis")
user_input = input("Enter a sentence:")
result = Classifier(user_input)
print(result)
