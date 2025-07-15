# train.py
import pandas as pd
from datasets import Dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

# 0. Cek dan tampilkan device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Menggunakan device:", device)

# 1. Load dataset
df = pd.read_csv("email_spam_indo.csv")[["Kategori", "Pesan"]]
df.columns = ["label", "text"]
df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
df = df.dropna(subset=["label", "text"])

# 2. Split train/test
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

# 3. Pastikan label int & reset index
train_df["label"] = train_df["label"].astype(int)
test_df["label"] = test_df["label"].astype(int)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# 4. Definisikan fitur eksplisit
features = Features({
    "text": Value("string"),
    "label": ClassLabel(num_classes=2, names=["ham", "spam"])
})

# 5. Buat HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df, features=features)
test_dataset = Dataset.from_pandas(test_df, features=features)

# 6. Load tokenizer IndoBERT
tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

def tokenize_and_format(example):
    tokens = tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)
    tokens["label"] = example["label"]
    return tokens

train_dataset = train_dataset.map(tokenize_and_format)
test_dataset = test_dataset.map(tokenize_and_format)

# 7. Load IndoBERT model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    "indobenchmark/indobert-base-p1",
    num_labels=2
).to(device)  # ⬅️ Kirim model ke GPU jika tersedia

# 8. Evaluasi metrics
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds)
    }

# 9. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    report_to="none"  # hilangkan warning wandb
)

# 10. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 11. Mulai training
trainer.train()

# 12. Simpan model & tokenizer
model.save_pretrained("indobert_spam_model")
tokenizer.save_pretrained("indobert_spam_tokenizer")

print("✅ Model & tokenizer berhasil disimpan")
