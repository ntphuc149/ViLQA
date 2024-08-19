from transformers import TrainingArguments

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

MODEL_CHECKPOINT = "google-bert/bert-base-multilingual-cased"
MAX_LENGTH = 256
STRIDE = 50
N_BEST = 50
MAX_ANSWER_LENGTH = 512

WANDB_TOKEN = 'YOUR_WANDB_TOKEN'
HF_ACCESS_TOKEN = 'YOUR_HF_ACCESS_TOKEN'

TRAINING_ARGS = TrainingArguments(
    "your-folder-location-name",
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    num_train_epochs=100,
    weight_decay=0.01,
    fp16=True,
)
