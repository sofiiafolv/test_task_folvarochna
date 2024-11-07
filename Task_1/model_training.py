from transformers import BertTokenizer, BertForTokenClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, random_split 
from tqdm import tqdm 
import torch
import json

# Load the pre-trained BERT model and tokenizer
entity_types = ['0', 'B-MOUNTAIN', 'I-MOUNTAIN']

model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(entity_types))
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the dataset
with open('strings.json') as f:
    data = json.load(f)

# Tokenize the dataset
def tokenize_and_align_labels(data, tokenizer):
    tokenized_data = []
    for example in data:
        tokenized_inputs = tokenizer(example["words"], truncation=True, is_split_into_words=True)
        for i, label in enumerate(example["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] + 1)
                previous_word_idx = word_idx

            padding_length = tokenizer.model_max_length - len(label_ids)
            word_ids = word_ids + tokenizer.pad_token_id * padding_length
            label_ids = label_ids + ([-100] * padding_length)
            tokenized_data.append({"input_ids": word_ids, "labels": label_ids})

    dataset = TensorDataset(
        torch.tensor([example["input_ids"] for example in tokenized_data]),
        torch.tensor([example["labels"] for example in tokenized_data]),
    )

    return dataset

train_dataset = tokenize_and_align_labels(data, tokenizer)

# Define parameters for the training
batch_size = 8
learning_rate = 2e-5
epochs = 15
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Create a DataLoader for the training and validation sets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train the model
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader):
        inputs, labels = batch
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_ner_model')