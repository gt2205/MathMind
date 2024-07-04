

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q peft transformers datasets accelerate peft

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q -i  https://pypi.org/simple/ bitsandbytes

from typing import Dict, List
from datasets import Dataset, load_dataset, disable_caching
disable_caching()
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset
from IPython.display import Markdown

dataset = load_dataset("microsoft/orca-math-word-problems-200k" , split = 'train')
small_dataset = dataset.select([i for i in range(200)])
print(small_dataset)
print(small_dataset[0])


prompt_template = """Below is an mathematics word problem. Solve the problem and give answer to it. question: {question}\n answer:"""
answer_template = """{answer}"""


def _add_text(rec):
    instruction = rec["question"]
    response = rec["answer"]

    if not instruction:
        raise ValueError(f"Expected an instruction in: {rec}")
    if not response:
        raise ValueError(f"Expected a response in: {rec}")
    rec["prompt"] = prompt_template.format(question=instruction)
    rec["answer"] = answer_template.format(answer=response)
    rec["text"] = rec["prompt"] + rec["answer"]
    return rec


small_dataset = small_dataset.map(_add_text)
print(small_dataset[0])

from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model_id = "databricks/dolly-v2-3b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # use_cache=False,
    device_map="auto",
    quantization_config=config
)


model.resize_token_embeddings(len(tokenizer))

from functools import partial
import copy
from transformers import DataCollatorForSeq2Seq

MAX_LENGTH = 256


def _preprocess_batch(batch: Dict[str, List]):
    model_inputs = tokenizer(batch["text"], max_length=MAX_LENGTH, truncation=True, padding='max_length')
    model_inputs["labels"] = copy.deepcopy(model_inputs['input_ids'])
    return model_inputs

_preprocessing_function = partial(_preprocess_batch)


encoded_small_dataset = small_dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["question", "prompt", "answer"],
)
processed_dataset = encoded_small_dataset.filter(lambda rec: len(rec["input_ids"]) <= MAX_LENGTH)


split_dataset = processed_dataset.train_test_split(test_size=14, seed=0)
print(split_dataset)


data_collator = DataCollatorForSeq2Seq(
        model = model, tokenizer=tokenizer, max_length=MAX_LENGTH, pad_to_multiple_of=8, padding='max_length')

from peft import LoraConfig, get_peft_model
from peft import prepare_model_for_kbit_training

LORA_R = 128
LORA_ALPHA = 512
LORA_DROPOUT = 0.05

lora_config = LoraConfig(
                 r = LORA_R,
                 lora_alpha = LORA_ALPHA,
                 lora_dropout = LORA_DROPOUT,
                 bias="none",
                 task_type="CAUSAL_LM",
                 target_modules=["query_key_value"],
)




model = prepare_model_for_kbit_training(model)


model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import TrainingArguments, Trainer
import bitsandbytes

EPOCHS = 2
LEARNING_RATE = 1e-4
MODEL_SAVE_FOLDER_NAME = "dolly-3b-lora"
training_args = TrainingArguments(
                    output_dir=MODEL_SAVE_FOLDER_NAME,
                    overwrite_output_dir=True,
                    fp16=True,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    learning_rate=LEARNING_RATE,
                    num_train_epochs=EPOCHS,
                    logging_strategy="epoch",
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
)
# training the model
trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset['train'],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
)
model.config.use_cache = False
trainer.train()

trainer.model.save_pretrained(MODEL_SAVE_FOLDER_NAME)

trainer.save_model(MODEL_SAVE_FOLDER_NAME)
trainer.model.config.save_pretrained(MODEL_SAVE_FOLDER_NAME)

def postprocess(response):
    messages = response.split("answer:")
    if not messages:
        raise ValueError("Invalid template for prompt. The template should include the term 'Response:'")
    return "".join(messages[1:])

inference_prompt = "Jungkook is the 5th place. Find the number of people who crossed the finish line faster than Jungkook."

inf_pipeline =  pipeline('text-generation', model=trainer.model, tokenizer=tokenizer, max_length=256, trust_remote_code=True)

response = inf_pipeline(prompt_template.format(question=inference_prompt))[0]['generated_text']

formatted_response = postprocess(response)
formatted_response
