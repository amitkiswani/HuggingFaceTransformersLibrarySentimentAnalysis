# SentimentAnalysis using HuggingFace TransformersLibrary 
Transformers library for sentiment analysis tasks

Python script that demonstrates how to use the Transformers library for sentiment analysis tasks. Here's a breakdown of the different sections:

**1. Transformers installation:**
Install the Transformers library using pip.

**2. Sentiment analysis with pipeline:**
Create a sentiment analysis pipeline and uses it to analyze different sentences.
It demonstrates using two different pre-trained sentiment analysis models: nlptown/bert-base-multilingual-uncased-sentiment and distilbert-base-uncased-finetuned-sst-2-english.
It shows how to get the predicted sentiment labels and their scores.

**3. Fine-tuning a model for sentiment analysis:**
Load a pre-trained DistilBert model and creates a tokenizer for it.
It then demonstrates how to prepare input data for the model using the tokenizer.
It performs sentiment analysis on multiple sentences and prints the predicted labels and scores.

**4. Saving and loading model and tokenizer:**
Show how to save both the model and tokenizer to a specific directory.
It then demonstrates how to load them back from the saved directory.

**5. Exploring model outputs:**
Perform sentiment analysis on multiple sentences with labels provided (positive and negative).
It calculates softmax probabilities for the predictions.

**6. Using DistilBert with custom configuration:**
Use DistilBert with a custom configuration, changing the number of heads, dimensions, and hidden dimensions.

**7. Creating a model from scratch:**
How to create a DistilBert model from scratch with a specific number of labels.

Overall, this code provides a comprehensive example of how to use the Transformers library for sentiment analysis tasks. It covers various aspects, including using pre-trained models, fine-tuning models, saving and loading models, and exploring model outputs.
