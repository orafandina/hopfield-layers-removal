

from transformers import TrainingArguments, Trainer



def evaluate_model(model, data, tokenizer, compute_metrics):
    args=TrainingArguments("test_trainer")
    score=Trainer(model=model, args=args, tokenizer=tokenizer, eval_dataset=data, compute_metrics=compute_metrics)
    result_score=score.evaluate()
    return result_score