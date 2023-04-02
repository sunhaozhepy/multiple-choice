from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset, DatasetDict, concatenate_datasets
import random

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=512)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

cnn = load_dataset("cnn_dailymail", "3.0.0")
xsum = load_dataset("xsum")

summary_queries = [
    "what is the theme of this article?",
    "how would you summarize this article?",
    "what is this article mainly about?",
    "this article can be summarized as _.",
    "what do we know as described in this article?",
    "what happened according to the article?",
    "summarization: _.",
    "what is the main idea of the article?",
    "what is the main plot of this article?",
    "what does the author try to convey in brief?"
]

def cnn_transform(examples):
  inputs = tokenizer([summary_queries[random.randint(0, len(summary_queries) - 1)] + "</s>" + s.replace("(CNN)", '').lower() for s in examples["article"]], truncation=True)
  labels = tokenizer([s.lower() for s in examples["highlights"]], truncation=True)

  return {
      "input_ids": inputs["input_ids"],
      "labels": labels["input_ids"]
  }

def xsum_transform(examples):
  inputs = tokenizer([summary_queries[random.randint(0, len(summary_queries) - 1)] + "</s>" + s.lower() for s in examples["document"]], truncation=True)
  labels = tokenizer([s.lower() for s in examples["summary"]], truncation=True)

  return {
      "input_ids": inputs["input_ids"],
      "labels": labels["input_ids"]
  }

cnn_dataset = DatasetDict(
  train=cnn["train"].map(cnn_transform, remove_columns=["article", "highlights", "id"], batched=True), 
  val=cnn['validation'].map(cnn_transform, remove_columns=["article", "highlights", "id"], batched=True), 
  test=cnn["test"].map(cnn_transform, remove_columns=["article", "highlights", "id"], batched=True)
)

xsum_dataset = DatasetDict(
  train=xsum["train"].map(xsum_transform, remove_columns=["document", "summary", "id"], batched=True), 
  val=xsum['validation'].map(xsum_transform, remove_columns=["document", "summary", "id"], batched=True), 
  test=xsum["test"].map(xsum_transform, remove_columns=["document", "summary", "id"], batched=True)
)

dataset = DatasetDict(
    train=concatenate_datasets([cnn_dataset["train"], xsum_dataset["train"]]),
    val=concatenate_datasets([cnn_dataset["val"], xsum_dataset["val"]]),
    test=concatenate_datasets([cnn_dataset["test"], xsum_dataset["test"]])
)
dataset.set_format("torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

arguments = Seq2SeqTrainingArguments(
    output_dir="../autodl-tmp/pretrain-3e4-checkpoints",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    learning_rate=3e-4,
    weight_decay=0.01,
    max_grad_norm=5.0,
    fp16=True,
    save_strategy="epoch",
    warmup_ratio=0.01,
    dataloader_num_workers=16,
    group_by_length=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset['val'],
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()