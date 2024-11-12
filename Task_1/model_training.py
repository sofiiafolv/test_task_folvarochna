import json
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True) # tokenize all words

    labels = []
    for i, label in enumerate(examples[f"ner_tags"]): # aligning labels with tokenized words
        word_ids = tokenized_inputs.word_ids(batch_index=i) # mapping all tokens to their correspondig word
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100) # assignning label -100 to all special tokens
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx]) # labeling the first token of given word
            else:
                label_ids.append(-100) # labeling the rest of tokens in the word
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    label_list = ["O", "B-MOUNTAIN", "I-MOUNTAIN"]

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2) # getting the most probable label for each token

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ] # removing special tokens from predictions
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ] # removing special tokens from labels

    results = seqeval.compute(predictions=true_predictions, references=true_labels) 
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    } # computing precision, recall, f1 and accuracy to evaluate the model

if __name__ == '__main__':
    with open('mountain_ner_dataset.json') as f:
        data = json.load(f)

    dataset = Dataset.from_dict({
        'sentence': [entry['sentence'] for entry in data],
        'tokens': [entry['tokens'] for entry in data],
        'ner_tags': [entry['ner_tags'] for entry in data],
    }) # loading the dataset in suitable format

    # data preprocessing
    train_test_split = dataset.train_test_split(test_size=0.1, shuffle=True) 

    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize_and_align_labels, batched=True) 

    id2label = {
        0: "O",
        1: "B-MOUNTAIN",
        2: "I-MOUNTAIN",
    }
    label2id = {
        "O": 0,
        "B-MOUNTAIN": 1,
        "I-MOUNTAIN": 2,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id
    ).to(device) # loading the model

    seqeval = evaluate.load("seqeval")

    training_args = TrainingArguments(
        output_dir="mountain_ner_model",
        learning_rate=1e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save(model.state_dict(), "model_weights.pth")
