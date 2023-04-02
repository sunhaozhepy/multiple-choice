import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import DatasetDict, load_dataset

device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

dataset = load_dataset("race", "all")

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", model_max_length=512)

answer_dict = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3
}

# for validation and test sets, we don't split the distractors; we use them as golden answers to compute the BLEU and ROUGE score
def no_split(example):
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

  return {
      "inputs": example["question"].lower() + "</s>" + answer.lower() + "</s>" + example["article"].lower(),
      "labels": [i.lower() for i in example["options"]]
  }

split_dataset = DatasetDict(
    val=dataset["validation"].map(no_split, remove_columns=['example_id', 'article', 'answer', 'question', 'options']),
    test=dataset["test"].map(no_split, remove_columns=['example_id', 'article', 'answer', 'question', 'options']),
)

dataloader = DataLoader(split_dataset["val"], batch_size=8, shuffle=False)

model = BartForConditionalGeneration.from_pretrained(f"../autodl-tmp/checkpoints/checkpoint-20590").to(device)
predictions = []

model.eval()
for i, batch in enumerate(dataloader):
  if i % 10 == 0:
    print(f"batch {i}...")
  inputs = batch["inputs"]
  model_inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
  with torch.no_grad():
    output_sequences = model.generate(**model_inputs, num_beams=12, num_beam_groups=3, diversity_penalty=1.0, num_return_sequences=3)
  predictions += [tokenizer.batch_decode(output_sequences, skip_special_tokens=True)]

print(predictions)