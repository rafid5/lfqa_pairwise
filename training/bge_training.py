
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


class PairwiseRerankerTrainer:
    """
    Fine-tune a cross-encoder reranker on pairwise human preference data.

    Each JSON line in train/dev/test is expected to have:
      - question_text
      - answer_1
      - answer_2
      - human_judgment: "answer_1" or "answer_2"
    """

    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        max_length: int = 1024,
        output_dir: str = "./bge_reranker_lfqa",
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Single regression-style logit as relevance score
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=1,
        )

        self.train_samples: List[Dict] = []
        self.dev_samples: List[Dict] = []
        self.test_samples: List[Dict] = []

        self.train_dataset = None
        self.dev_dataset = None

    # ---------- Data utilities ----------

    @staticmethod
    def load_jsonl(path: str) -> List[Dict]:
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
        """
        Convert each pairwise question into two pointwise items:
          - (q, preferred_answer, label=1)
          - (q, other_answer, label=0)
        """
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

            # positive pair
            rows.append({"query": q, "doc": pos, "label": 1.0})
            # negative pair
            rows.append({"query": q, "doc": neg, "label": 0.0})

        return rows

    def load_all_splits(self) -> None:
        raw_train = self.load_jsonl(self.train_path)
        raw_dev   = self.load_jsonl(self.dev_path)
        raw_test  = self.load_jsonl(self.test_path)

        # Keep only samples where human_judgment is NOT 'tie'
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
        """
        Build HuggingFace Datasets for train/dev with tokenization.
        """

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

    # ---------- Training ----------

    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        # logits: (N, 1)
        scores = logits.squeeze(-1)
        labels = labels.astype(int)

        # Threshold at 0.0 (compatible with BCEWithLogitsLoss)
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

    def train(self, num_train_epochs: int = 2, lr: float = 2e-5, batch_size: int = 2):
        use_fp16 = torch.cuda.is_available()

        # Older transformers: keep only the basic, supported arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            num_train_epochs=num_train_epochs,
            logging_steps=50,
            fp16=use_fp16,  # if this causes an error, just delete this line
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,   # used when you call trainer.evaluate()
            compute_metrics=self.compute_metrics,
        )

        # Train on train_dataset
        trainer.train()

        # Optional: evaluate once on the dev set at the end
        print("\n=== Dev set evaluation (end of training) ===")
        dev_metrics = trainer.evaluate()
        for k, v in dev_metrics.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

        # Save final model + tokenizer
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


    # ---------- Evaluation at the pairwise level ----------

    def evaluate_pairwise(self) -> Tuple[float, float, float, float]:
        """
        Evaluate on original pairwise task:
          For each question, does the model pick the same answer
          as human_judgment?

        We compute accuracy, precision, recall, F1 with "answer_2"
        treated as the positive class (1).
        """
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
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

                # Encode "answer_2" as positive class (1)
                y_pred.append(1 if pred_label == "answer_2" else 0)
                y_true.append(1 if true_label == "answer_2" else 0)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        print("\n=== Pairwise Evaluation on Test Set ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

        return acc, precision, recall, f1


def main():
    # 🔁 Change these paths to your actual files
    train_path = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_train.jsonl"
    dev_path   = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_dev.jsonl"
    test_path  = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_test.jsonl"

    trainer = PairwiseRerankerTrainer(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        model_name="BAAI/bge-reranker-v2-m3",
        max_length=1024,           # long(er) context, no manual chunking
        output_dir="./bge_reranker_lfqa",
    )

    # 1) Load data from the 3 jsonl files
    trainer.load_all_splits()

    # 2) Build tokenized HF datasets
    trainer.prepare_datasets()

    # 3) Train
    trainer.train(
        num_train_epochs=2,
        lr=2e-5,
        batch_size=2,  # adjust if you have more VRAM
    )

    # 4) Evaluate on pairwise test set
    trainer.evaluate_pairwise()


if __name__ == "__main__":
    main()
