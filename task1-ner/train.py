import numpy as np
import torch
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mountain NER Model")
    parser.add_argument("--model_name", type=str, default="distilbert-base-cased", help="Base model name or path")
    parser.add_argument("--dataset_path", type=str, default="data/ner_mountain_dataset.json", help="Path to JSON dataset")
    parser.add_argument("--output_dir", type=str, default="model", help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    return parser.parse_args()


def train_ner():
    args = parse_args()
    set_seed(42)

    # Load Local JSON Dataset
    print(f"Loading local JSON dataset from: {args.dataset_path}")
    dataset = load_dataset("json", data_files=args.dataset_path)

    if "validation" not in dataset and "test" not in dataset:
        print("Splitting dataset into train and validation sets...")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    eval_split_name = "validation" if "validation" in dataset else "test"
    print(f"Using '{eval_split_name}' split for evaluation.")

    # Extract String Labels Automatically
    print("Extracting unique string labels from dataset...")
    unique_labels = set()
    for example in dataset["train"]:
        unique_labels.update(example["tags"])

    label_list = sorted(list(unique_labels))
    if "O" in label_list:
        label_list.remove("O")
        label_list = ["O"] + label_list

    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    print(f"Detected {num_labels} labels: {label_list}")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Tokenization & Label Alignment Function
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=256
        )

        labels = []
        for i, label_seq in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens ([CLS], [SEP], [PAD]) get -100
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    string_tag = label_seq[word_idx]
                    label_ids.append(label2id[string_tag])
                else:
                    string_tag = label_seq[word_idx]
                    if string_tag.startswith("B-"):
                        string_tag = "I-" + string_tag[2:]
                    label_ids.append(label2id[string_tag])

                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    print("Tokenizing and aligning labels...")
    tokenized_datasets = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # Data Collator and Metrics Setup
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_preds = []
        true_labels = []
        for batch_pred, batch_label in zip(predictions, labels):
            for p_idx, l_idx in zip(batch_pred, batch_label):
                if l_idx != -100:
                    true_preds.append(label_list[p_idx])
                    true_labels.append(label_list[l_idx])

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels,
            true_preds,
            average="weighted",
            zero_division=0
        )
        acc = accuracy_score(true_labels, true_preds)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall
        }

    # Training Arguments & Trainer Setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        fp16=torch.cuda.is_available(),
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        push_to_hub=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets[eval_split_name],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    model.config.id2label = id2label
    model.config.label2id = label2id
    model.config.save_pretrained(args.output_dir)


if __name__ == "__main__":
    train_ner()