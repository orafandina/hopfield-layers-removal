from transformers import TrainingArguments, Trainer
import numpy as np
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AlbertForSequenceClassification, AlbertConfig, AlbertTokenizer
from datasets import load_dataset
import torch
from evaluate import load

path_to_finetuned="C:/Users/hopfiled-based-layers-removal/"
tuning_params={
    'evaluation_strategy': 'epoch',
    'save_strategy': 'epoch',
    'learning_rate': 2e-5,
    'batch_size': 16,
    'num_train_epochs': 5,
    'weight_decay': 0.01,
    'load_best_model_at_end': True,
}

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#for each task: finetune the model
for task in GLUE_TASKS:
    actual_task = "mnli" if task == "mnli-mm" else task
    metric=load('glue', actual_task)
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    dataset = load_dataset("glue", actual_task)
    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"


    model_checkpoint = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    model=model.to(device)


    tokenizer_checkpoint="bert-base-uncased"  #"albert-large-v2" #24*16 #"albert-base-v2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint, truncation=True)

    #preprocess the data: tokenize the relevant pieces
    sentence1_key, sentence2_key = task_to_keys[actual_task]
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    train_dataset=encoded_dataset['train']
    eval_dataset=encoded_dataset[validation_key]



    #fine-tuning the model
    model_name = model_checkpoint.split("/")[-1]
    finetuned_name=f"{model_name}-finetuned-{task}"
    #print(finetuned_name)


    args = TrainingArguments(
            f"{model_name}-finetuned-{task}",
            evaluation_strategy = tuning_params['evaluation_strategy'],
            save_strategy = tuning_params['save_strategy'],
            learning_rate=2e-5,
            per_device_train_batch_size=tuning_params['batch_size'],
            per_device_eval_batch_size=tuning_params['batch_size'],
            num_train_epochs=tuning_params['num_train_epochs'],
            weight_decay=tuning_params['weight_decay'],
            load_best_model_at_end=tuning_params['load_best_model_at_end'],
            metric_for_best_model=metric_name,
            report_to="none",)

    trainer=Trainer(
            model,
            args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics)


    trainer.train()

    trainer.save_model(finetuned_name)

    model_save_name = f"{model_name}-finetuned-{task}"
    path = f"{path_to_finetuned}{model_save_name}"
    torch.save(model.state_dict(), path)