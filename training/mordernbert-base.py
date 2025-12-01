import json
from typing import List, Dict, Tuple

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


class ModernBERTRerankerTrainer:


    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        model_name: str = "answerdotai/ModernBERT-base",
        max_length: int = 4096,  # long context
        output_dir: str = "./modernbert_reranker_lfqa",
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir

        # Load tokenizer
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading ModernBERT model for sequence classification: {self.model_name}")
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
        )
        # Initialize data containers
        self.train_samples: List[Dict] = []
        self.dev_samples: List[Dict] = []
        self.test_samples: List[Dict] = []

        self.train_dataset = None
        self.dev_dataset = None


    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
        # Load a JSONL file into a list of dicts.
        data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data

    @staticmethod
    def expand_pointwise(samples: List[Dict]) -> List[Dict]:
        # Convert pairwise samples into pointwise format.
        rows = []
        for ex in samples:
            q = ex["question_text"]
            a1 = ex["answer_1"]
            a2 = ex["answer_2"]
            pref = ex["human_judgment"]

            if pref == "answer_1":
                pos, neg = a1, a2
            else:
                pos, neg = a2, a1

            rows.append({"query": q, "doc": pos, "label": 1.0})
            rows.append({"query": q, "doc": neg, "label": 0.0})

        return rows

    def load_all_splits(self) -> None:
        # Load and filter all data splits.
        raw_train = self.load_jsonl(self.train_path)
        raw_dev   = self.load_jsonl(self.dev_path)
        raw_test  = self.load_jsonl(self.test_path)

      
        self.train_samples = [
            ex for ex in raw_train
            if ex.get("human_judgment") in ("answer_1", "answer_2")
        ]
        self.dev_samples = [
            ex for ex in raw_dev
            if ex.get("human_judgment") in ("answer_1", "answer_2")
        ]
        self.test_samples = [
            ex for ex in raw_test
            if ex.get("human_judgment") in ("answer_1", "answer_2")
        ]

        print(f"Train samples (no tie): {len(self.train_samples)} / {len(raw_train)}")
        print(f"Dev samples   (no tie): {len(self.dev_samples)} / {len(raw_dev)}")
        print(f"Test samples  (no tie): {len(self.test_samples)} / {len(raw_test)}")

    def prepare_datasets(self) -> None:

        # Prepare datasets for training and evaluation.
        def tokenize_batch(batch):
            enc = self.tokenizer(
                batch["query"],
                batch["doc"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            enc["labels"] = batch["label"]
            return enc

        train_rows = self.expand_pointwise(self.train_samples)
        dev_rows = self.expand_pointwise(self.dev_samples)

        train_ds = Dataset.from_list(train_rows)
        dev_ds = Dataset.from_list(dev_rows)

        train_ds = train_ds.map(
            tokenize_batch, batched=True, remove_columns=train_ds.column_names
        )
        dev_ds = dev_ds.map(
            tokenize_batch, batched=True, remove_columns=dev_ds.column_names
        )

        train_ds.set_format(type="torch")
        dev_ds.set_format(type="torch")

        self.train_dataset = train_ds
        self.dev_dataset = dev_ds



    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        # Compute evaluation metrics.
        logits, labels = eval_pred
       
        scores = logits.squeeze(-1)
        labels = labels.astype(int)

        
        preds = (scores > 0.0).astype(int)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train(self, num_train_epochs: int = 1, lr: float = 5e-5, batch_size: int = 1):
        # Train the model.
        use_fp16 = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_steps=20,
            fp16=use_fp16,
        )
        # Initialize Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        # Evaluate on dev set
        print("\n=== Dev set evaluation (end of training) ===")
        dev_metrics = trainer.evaluate()
        for k, v in dev_metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)



    def evaluate_pairwise(self) -> Tuple[float, float, float, float]:
        # Evaluate the model on the test set in a pairwise manner.
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Evaluation device:", device)
        self.model.to(device)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for ex in self.test_samples:
                q = ex["question_text"]
                a1 = ex["answer_1"]
                a2 = ex["answer_2"]

                inputs_1 = self.tokenizer(
                    q,
                    a1,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                ).to(device)

                inputs_2 = self.tokenizer(
                    q,
                    a2,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                ).to(device)

                score_1 = self.model(**inputs_1).logits.squeeze(-1).item()
                score_2 = self.model(**inputs_2).logits.squeeze(-1).item()

                pred_label = "answer_1" if score_1 > score_2 else "answer_2"
                true_label = ex["human_judgment"]

                
                y_pred.append(1 if pred_label == "answer_2" else 0)
                y_true.append(1 if true_label == "answer_2" else 0)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        # Print results
        print("\n=== Pairwise Evaluation on Test Set ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        return acc, precision, recall, f1


def main():
    # Define file paths
    train_path = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_train.jsonl"
    dev_path   = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_dev.jsonl"
    test_path  = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_test.jsonl"
    # Initialize trainer
    trainer = ModernBERTRerankerTrainer(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        model_name="answerdotai/ModernBERT-base",
        max_length=4096,  # long context
        output_dir="./modernbert_reranker_lfqa",
    )
    # Load data and prepare datasets
    trainer.load_all_splits()
    trainer.prepare_datasets()
    # Train the model
    trainer.train(
        num_train_epochs=1,
        lr=5e-5,
        batch_size=1, 
    )
    # Evaluate on test set
    trainer.evaluate_pairwise()


if __name__ == "__main__":
    main()
