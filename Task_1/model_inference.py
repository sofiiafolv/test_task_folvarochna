import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import argparse

def classify(tokenizer, model, text):
    classifier = pipeline("ner", model=model, tokenizer=tokenizer)
    predictions = classifier(text)

    mountain_names = []
    current_mountain = []

    for item in predictions:
        token = text[item['start']:item['end']]
        entity_type = item['entity']
        
        if entity_type == "B-MOUNTAIN":
            if current_mountain:
                mountain_names.append(" ".join(current_mountain))
            current_mountain = [token] 

        elif entity_type == "I-MOUNTAIN" and current_mountain:
            current_mountain.append(token)
        
        elif current_mountain:
            mountain_names.append(" ".join(current_mountain))
            current_mountain = []

    if current_mountain:
        mountain_names.append(" ".join(current_mountain))

    return mountain_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str, help="Path to the text file with sentences.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
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
    )

    model.load_state_dict(torch.load("model_weights.pth"))

    with open(args.file_path, "r") as file:
        text = file.read()

    print("Input text:", text)

    mountain_names = classify(tokenizer, model, text)
    print("Identified mountain names:")
    for mountain in mountain_names:
        print(mountain)
