import os

from datasets import load_dataset, load_metric, Dataset
import pandas as pd
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LongformerModel,
)

# load rouge
rouge = load_metric("rouge")

EXPORT_PATH = "../data/baseline_models/t5_finetuned"
os.makedirs(EXPORT_PATH, exist_ok=False)

train_df = pd.read_json("../data/final_train.json")
val_df = pd.read_json("../data/final_dev.json")

train_df['prepared_input'] = train_df.apply(lambda row: f"question: {row['title']}  context: {row['text']} </s>", axis=1)
val_df['prepared_input'] = val_df.apply(lambda row: f"question: {row['title']}  context: {row['text']} </s>", axis=1)

x_col = "prepared_input"
label_col = "answer"

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-base-squad")

# Not enough GPU memory? Reduce batch size or if absolutely neccessary encoder_max_length
encoder_max_length = 4096 #2048
decoder_max_length = 512
batch_size = 4 #1

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(
        batch[x_col],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
    )
    outputs = tokenizer(
        batch[label_col],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask

    # create 0 global_attention_mask lists
    batch["global_attention_mask"] = len(batch["input_ids"]) * [
        [0 for _ in range(len(batch["input_ids"][0]))]
    ]

    # since above lists are references, the following line changes the 0 index for all samples
    batch["global_attention_mask"][0][0] = 1
    batch["labels"] = outputs.input_ids

    # We have to make sure that the PAD token is ignored
    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


# map train data
train_dataset = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=[x_col, label_col],
)

# map val data
val_dataset = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
    remove_columns=[x_col, label_col],
)

# set Python list to PyTorch tensor
train_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# set Python list to PyTorch tensor
val_dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "global_attention_mask", "labels"],
)

# enable fp16 apex training
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    #fp16_backend="apex",
    output_dir=EXPORT_PATH,
    overwrite_output_dir=True,
    logging_steps=250,
    eval_steps=5000,
    save_steps=500,
    warmup_steps=1500,
    save_total_limit=2,
    gradient_accumulation_steps=4,
)


# compute Rouge score during validation
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(
        predictions=pred_str, references=label_str, rouge_types=["rouge2"]
    )["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


# load model + enable gradient checkpointing & disable cache for checkpointing
#led = LongformerModel.from_pretrained("valhalla/t5-base-squad", gradient_checkpointing=True, use_cache=False)
led = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-base-squad", use_cache=False)

# set generate hyperparameters
led.config.num_beams = 4
led.config.max_length = 512
led.config.min_length = 1
led.config.length_penalty = 2.0
led.config.early_stopping = True
led.config.no_repeat_ngram_size = 3


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=led,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# start training
trainer.train()

#Change this path to output folder for binary model!
trainer.save_model(EXPORT_PATH)
