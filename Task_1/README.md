# Task 1. Natural Language Processing. Named entity recognition

## Solution explanation

To solve this task I trained DistilBERT base model from HuggingFace. I used pipeline in [Token Classification article from HuggingFace.](https://huggingface.co/docs/transformers/v4.37.2/tasks/token_classification)

### Data preprocessing

All sentences are divided into words in training dataset, so I used tokenizer to tokenize words into subwords. because of special tokens and subwords, there is mismatch between words and labels, so I needed to align them. I used word_ids  method, and then set label -100 to special tokens and subtokens from the same word, and labeled with corresponding label the first part of a given word.

I used Kaggle to train model on GPU, so for that I also used DataCollatorForTokenClassification to dynamically pad the sentences to the longest length in a batch during collation. I also trained model on CPU on my local machine, as it didn;t take that much time.

### Model training

I loaded model from HuggingFace defined training hyperparameters in TrainingArguments and used Trainer from transformers library. Model is finetuned by calling train().

[Link to model_weights](https://drive.google.com/file/d/1K79fypVRTUAfto_qIi83kH7jkjdXXtFI/view?usp=sharing)

### Model evaluation

Despite loss, I also computed accuracy, precision, recall and f1 score, as they are more interpretable. Accuracy wasn;t enough as classes are really unbalanced. I used compute_metrics from eva;uate library.

## Project setup

All project setup is in "demo.ipynb":

This line of code downloads all necessary packages to run project.

``!pip install -r requirements.txt``

This code starts training the model and saves model weights in this folder.

``!python model_training.py``

This code runs Python script which performs inference to make predictions on text in a file "text_file.txt" which can be replaced with another path.

``!python model_inference.py text_file.txt``
