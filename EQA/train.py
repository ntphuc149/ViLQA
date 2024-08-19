from transformers import Trainer
from data.data_processing import read_dataset, prepare_dataset, preprocess_training_examples, preprocess_validation_test_examples
from models.eqa_model import load_model_and_tokenizer
from config import MODEL_CHECKPOINT, MAX_LENGTH, STRIDE, TRAINING_ARGS, N_BEST, MAX_ANSWER_LENGTH, WANDB_TOKEN, HF_ACCESS_TOKEN
from utils.metrics import compute_metrics

def main():
    

    df = read_dataset(r'./data/dataset/ALQAC.csv')
    train_set, val_set, test_set = prepare_dataset(df)
    
    model, tokenizer = load_model_and_tokenizer(MODEL_CHECKPOINT)
    
    train_dataset = train_set.map(
        lambda examples: preprocess_training_examples(examples, tokenizer, MAX_LENGTH, STRIDE),
        batched=True,
        remove_columns=train_set.column_names,
    )

    val_dataset = val_set.map(
        lambda examples: preprocess_validation_test_examples(examples, tokenizer, MAX_LENGTH, STRIDE),
        batched=True,
        remove_columns=train_set.column_names,
    )

    test_dataset = test_set.map(
        lambda examples: preprocess_validation_test_examples(examples, tokenizer, MAX_LENGTH, STRIDE),
        batched=True,
        remove_columns=train_set.column_names,
    )
    
    trainer = Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    trainer.save_model(r'./output')

    predictions, _, _ = trainer.predict(test_dataset)
    start_logits, end_logits = predictions
    
    results = compute_metrics(start_logits, end_logits, test_dataset, test_set, N_BEST, MAX_ANSWER_LENGTH)
    print(results)

if __name__ == "__main__":
    main()