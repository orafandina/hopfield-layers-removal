#Simple greedy algorithm over all tasks
#Greedy experiment: looping over all tasks, over all layers, removing one-by one and evaluating on the full test
#choosing the 5 layers with the min scores; print the layers


import numpy as np
import csv
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AlbertTokenizer, AlbertModel
from datasets import load_dataset, load_metric
import torch
from datasets import load_dataset
from evaluate import load
from  layers_removal import deleteEncodingLayers,substituteLayers,inverse_layers
from utils import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


#num layers to find
k=5

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb"]

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
    "wnli": ("sentence1", "sentence2"),
}
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True, truncation=True)
#tokenizer_checkpoint="albert-base-v2" #"albert-large-v2" #24*16 #"albert-base-v2"  #"bert-base-uncased" #albert; deberta
#tokenizer = AlbertTokenizer.from_pretrained(tokenizer_checkpoint, truncation=True)

for task in GLUE_TASKS:
    actual_task = "mnli" if task == "mnli-mm" else task
    print('here')
    dataset = load_dataset("glue", actual_task)
    metric = load('glue', actual_task)
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    model_checkpoint=f"orafandina/bert-base-uncased-finetuned-{task}"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
    model.to(device)
    #eval_dataset=dataset['validation']
    labeled_dataset=dataset['train']
    labeled_dataset=labeled_dataset.train_test_split(test_size=0.2, shuffle=True)
    #labeled_dataset['validation']=eval_dataset
    sentence1_key, sentence2_key = task_to_keys[actual_task]
    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = labeled_dataset.map(preprocess_function, batched=True)
    encoded_dataset.set_format("torch")
    train_dataset=encoded_dataset['train']
    test_dataset=encoded_dataset['test']
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)
    layers=[0,1,2,3,4,5,6,7,8,9,10,11]
    scores_arr=np.zeros(12)
    for layer in layers:
        peeled_model=deleteEncodingLayers(model, layers_to_keep=inverse_layers([layer]))
        test_score=evaluate_model(peeled_model, test_dataset, tokenizer, compute_metrics)
        print(f"Result for {task}, removed layer: {layer} is {test_score}")
        if(task=="cola"):
            scores_arr[layer]=test_score['eval_matthews_correlation']
        else:
            if(task=="stsb"):
                scores_arr[layer]=test_score['eval_pearson']
            else:
                scores_arr[layer]=test_score['eval_accuracy']
    print(scores_arr)

    indeces=np.argpartition(scores_arr, k)

    #remove the k greedy layers
    greedy_layers_to_remove=indeces[0:k].tolist()
    peeled_model=deleteEncodingLayers(model, layers_to_keep=inverse_layers(greedy_layers_to_remove))
    test_score=evaluate_model(peeled_model, test_dataset, tokenizer, compute_metrics)
    print(f"Greedy for {task}, removed layers: {greedy_layers_to_remove} is {test_score}")


