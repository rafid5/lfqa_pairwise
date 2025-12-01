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


class LongformerRerankerTrainer:
    """
    Fine-tune Longformer (long context) as a cross-encoder reranker
    on pairwise human preference data.

    Each JSON line in train/dev/test is expected to have:
      - question_text
      - answer_1
      - answer_2
      - human_judgment: "answer_1", "answer_2", or "tie"

    We expand each pair to two pointwise examples:
      - (question_text, preferred_answer, label=1)
      - (question_text, other_answer,    label=0)

    Then we evaluate at the pairwise level:
      - Does model(q, a1) vs model(q, a2) match human_judgment?
    """

    def __init__(
        self,
        train_path: str,
        dev_path: str,
        test_path: str,
        model_name: str = "allenai/longformer-base-4096",
        max_length: int = 4096,  # >1024, safe starting point for 8GB
        output_dir: str = "./longformer_reranker_lfqa",
    ):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.model_name = model_name
        self.max_length = max_length
        self.output_dir = output_dir

        # ---- Tokenizer & model ----
        print(f"Loading tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading Longformer model for sequence classification: {self.model_name}")
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
          - (q, other_answer,     label=0)
        Assumes 'human_judgment' is either 'answer_1' or 'answer_2'.
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

            rows.append({"query": q, "doc": pos, "label": 1.0})
            rows.append({"query": q, "doc": neg, "label": 0.0})

        return rows

    def load_all_splits(self) -> None:
        raw_train = self.load_jsonl(self.train_path)
        raw_dev   = self.load_jsonl(self.dev_path)
        raw_test  = self.load_jsonl(self.test_path)

        # Filter out 'tie'
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
        Longformer needs a global_attention_mask; we'll set CLS token as global.
        """

        def tokenize_batch(batch):
            enc = self.tokenizer(
                batch["query"],
                batch["doc"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
            )
            # Global attention mask: first token global, rest local
            global_attention_mask = []
            for mask in enc["attention_mask"]:
                g = [0] * len(mask)
                if len(g) > 0:
                    g[0] = 1  # CLS (first token) gets global attention
                global_attention_mask.append(g)

            enc["global_attention_mask"] = global_attention_mask
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

    def train(self, num_train_epochs: int = 1, lr: float = 5e-5, batch_size: int = 1):
        use_fp16 = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            logging_steps=20,
            fp16=use_fp16,  # if this crashes, set to False
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.dev_dataset,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()

        print("\n=== Dev set evaluation (end of training) ===")
        dev_metrics = trainer.evaluate()
        for k, v in dev_metrics.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")

        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

    # ---------- Pairwise Evaluation ----------

    def evaluate_pairwise(self) -> Tuple[float, float, float, float]:
        """
        Evaluate on original pairwise task:
          For each question, does the model pick the same answer
          as human_judgment?

        We compute accuracy, precision, recall, F1 with "answer_2"
        treated as the positive class (1), same as your BGE script.
        """
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

                # (q, a1)
                inputs_1 = self.tokenizer(
                    q,
                    a1,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                attn_1 = inputs_1["attention_mask"]
                global_attn_1 = torch.zeros_like(attn_1)
                global_attn_1[:, 0] = 1  # CLS token global
                inputs_1["global_attention_mask"] = global_attn_1
                inputs_1 = {k: v.to(device) for k, v in inputs_1.items()}

                # (q, a2)
                inputs_2 = self.tokenizer(
                    q,
                    a2,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                )
                attn_2 = inputs_2["attention_mask"]
                global_attn_2 = torch.zeros_like(attn_2)
                global_attn_2[:, 0] = 1
                inputs_2["global_attention_mask"] = global_attn_2
                inputs_2 = {k: v.to(device) for k, v in inputs_2.items()}

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
    # 🔁 Change to your actual files
    train_path = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_train.jsonl"
    dev_path   = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_dev.jsonl"
    test_path  = r"F:\PhD\Long form research question\Final Dataset\sample\LFQA-HP-1M_sample_test.jsonl"

    trainer = LongformerRerankerTrainer(
        train_path=train_path,
        dev_path=dev_path,
        test_path=test_path,
        model_name="allenai/longformer-base-4096",
        max_length=4096,  # >1024, adjust upward if VRAM allows
        output_dir="./longformer_reranker_lfqa",
    )

    trainer.load_all_splits()
    trainer.prepare_datasets()

    trainer.train(
        num_train_epochs=1,
        lr=5e-5,
        batch_size=1,  # start at 1 for safety
    )

    trainer.evaluate_pairwise()


if __name__ == "__main__":
    main()
