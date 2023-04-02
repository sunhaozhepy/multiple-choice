from transformers import BartTokenizer, BartForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=512)
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

dataset = load_dataset("race", "all")

answer_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

# we split distractors into three different training examples; this process takes about 10 minutes
def split1(example):
  # picking the answer
  answer_id = answer_dict[example["answer"]]
  answer = example["options"][answer_id]
  del example["options"][answer_id]
  # preprocessing the options: remove possible trailing blankspaces and periods at the end
  for i in range(len(example["options"])):
    example["options"][i] = example["options"][i].strip()
    if example["options"][i].endswith('.'):
      example["options"][i] = example["options"][i][:-1]
  # same thing with the answer
  answer = answer.strip()
  if answer.endswith('.'):
    answer = answer[:-1]
  # preprocessing the article: remove possible trailing blankspaces, and blankspaces before the period
  example["article"] = example["article"].strip()
  if example["article"].endswith('.'):
    while example["article"][-2] == ' ':
      example["article"] = example["article"][:-2] + example["article"][-1]
  # same thing with the question
  example["question"] = example["question"].strip()
  if not example["question"].endswith("?") and not example["question"].endswith('.'):
    example["question"] = example["question"] + '.'
  if example["question"].endswith('.'):
    while example["question"][-2] == ' ':
      example["question"] = example["question"][:-2] + example["question"][-1]
  # '_' should have one blankspace before and after it, unless it is followed by a period
  if '_' in example["question"]:
    index = example["question"].find('_')
    while example["question"][index-1] == ' ':
      example["question"] = example["question"][:index-1] + example["question"][index:]
      index = example["question"].find('_')
    index = example["question"].find('_')
    example["question"] = example["question"][:index] + ' ' + example["question"][index:]
    index = example["question"].find('_')
    if example["question"][index+1] != '.':
      while example["question"][index+1] == ' ':
        example["question"] = example["question"][:index+1] + example["question"][index+2:]
        index = example["question"].find('_')
      index = example["question"].find('_')
      example["question"] = example["question"][:index+1] + ' ' + example["question"][index+1:]
  inputs = tokenizer(example["question"].lower() + "</s>" + answer.lower() + "</s>" + example["article"].lower(), truncation=True)
  labels = tokenizer(example["options"][0].lower(), truncation=True)

  return {
      "input_ids": inputs["input_ids"],
      "labels": labels["input_ids"]
  }

def split2(example):
  # picking the answer
  answer_id = answer_dict[example["answer"]]
  answer = example["options"][answer_id]
  del example["options"][answer_id]
  # preprocessing the options: remove possible trailing blankspaces and periods at the end
  for i in range(len(example["options"])):
    example["options"][i] = example["options"][i].strip()
    if example["options"][i].endswith('.'):
      example["options"][i] = example["options"][i][:-1]
  # same thing with the answer
  answer = answer.strip()
  if answer.endswith('.'):
    answer = answer[:-1]
  # preprocessing the article: remove possible trailing blankspaces, and blankspaces before the period
  example["article"] = example["article"].strip()
  if example["article"].endswith('.'):
    while example["article"][-2] == ' ':
      example["article"] = example["article"][:-2] + example["article"][-1]
  # same thing with the question
  example["question"] = example["question"].strip()
  if not example["question"].endswith("?") and not example["question"].endswith('.'):
    example["question"] = example["question"] + '.'
  if example["question"].endswith('.'):
    while example["question"][-2] == ' ':
      example["question"] = example["question"][:-2] + example["question"][-1]
  # '_' should have one blankspace before and after it, unless it is followed by a period
  if '_' in example["question"]:
    index = example["question"].find('_')
    while example["question"][index-1] == ' ':
      example["question"] = example["question"][:index-1] + example["question"][index:]
      index = example["question"].find('_')
    index = example["question"].find('_')
    example["question"] = example["question"][:index] + ' ' + example["question"][index:]
    index = example["question"].find('_')
    if example["question"][index+1] != '.':
      while example["question"][index+1] == ' ':
        example["question"] = example["question"][:index+1] + example["question"][index+2:]
        index = example["question"].find('_')
      index = example["question"].find('_')
      example["question"] = example["question"][:index+1] + ' ' + example["question"][index+1:]
  inputs = tokenizer(example["question"].lower() + "</s>" + answer.lower() + "</s>" + example["article"].lower(), truncation=True)
  labels = tokenizer(example["options"][1].lower(), truncation=True)

  return {
      "input_ids": inputs["input_ids"],
      "labels": labels["input_ids"]
  }

def split3(example):
  # picking the answer
  answer_id = answer_dict[example["answer"]]
  answer = example["options"][answer_id]
  del example["options"][answer_id]
  # preprocessing the options: remove possible trailing blankspaces and periods at the end
  for i in range(len(example["options"])):
    example["options"][i] = example["options"][i].strip()
    if example["options"][i].endswith('.'):
      example["options"][i] = example["options"][i][:-1]
  # same thing with the answer
  answer = answer.strip()
  if answer.endswith('.'):
    answer = answer[:-1]
  # preprocessing the article: remove possible trailing blankspaces, and blankspaces before the period
  example["article"] = example["article"].strip()
  if example["article"].endswith('.'):
    while example["article"][-2] == ' ':
      example["article"] = example["article"][:-2] + example["article"][-1]
  # same thing with the question
  example["question"] = example["question"].strip()
  if not example["question"].endswith("?") and not example["question"].endswith('.'):
    example["question"] = example["question"] + '.'
  if example["question"].endswith('.'):
    while example["question"][-2] == ' ':
      example["question"] = example["question"][:-2] + example["question"][-1]
  # '_' should have one blankspace before and after it, unless it is followed by a period
  if '_' in example["question"]:
    index = example["question"].find('_')
    while example["question"][index-1] == ' ':
      example["question"] = example["question"][:index-1] + example["question"][index:]
      index = example["question"].find('_')
    index = example["question"].find('_')
    example["question"] = example["question"][:index] + ' ' + example["question"][index:]
    index = example["question"].find('_')
    if example["question"][index+1] != '.':
      while example["question"][index+1] == ' ':
        example["question"] = example["question"][:index+1] + example["question"][index+2:]
        index = example["question"].find('_')
      index = example["question"].find('_')
      example["question"] = example["question"][:index+1] + ' ' + example["question"][index+1:]
  inputs = tokenizer(example["question"].lower() + "</s>" + answer.lower() + "</s>" + example["article"].lower(), truncation=True)
  labels = tokenizer(example["options"][2].lower(), truncation=True)

  return {
      "input_ids": inputs["input_ids"],
      "labels": labels["input_ids"]
  }

split_dataset = concatenate_datasets([
  dataset["train"].filter(lambda example: len(example['question']) > 1).map(split1, remove_columns=['example_id', 'article', 'answer', 'question', 'options']), 
  dataset['train'].filter(lambda example: len(example['question']) > 1).map(split2, remove_columns=['example_id', 'article', 'answer', 'question', 'options']), 
  dataset["train"].filter(lambda example: len(example['question']) > 1).map(split3, remove_columns=['example_id', 'article', 'answer', 'question', 'options'])
])

split_dataset.set_format("torch")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

arguments = Seq2SeqTrainingArguments(
    output_dir="../autodl-tmp/checkpoints",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=5e-5,
    weight_decay=0.001,
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
    train_dataset=split_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()