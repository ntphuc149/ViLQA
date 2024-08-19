import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from config import TRAIN_RATIO, VAL_RATIO

def read_dataset(file_path):
    df = pd.read_csv(file_path)
    df['context'] = df['context'].astype(str)
    df['question'] = df['question'].astype(str)
    df['answer'] = df['answer'].astype(str)
    return df

def find_start_index(context, answer):
    return str(context).find(str(answer))


def prepare_dataset(df):
    df['start_index'] = df.apply(lambda row: find_start_index(context=row['context'], answer=row['answer']), axis=1)

    df = df[df['start_index'] != -1]

    dataset_temp = []
    for _, row in df.iterrows():
        sample = {
            'context': row['context'],
            'question': row['question'],
            'answer': {'text': [row['answer']], 'answer_start': [row['start_index']]}
        }
        dataset_temp.append(sample)

    dataset = pd.DataFrame(dataset_temp)

    num_of_total_sample = len(dataset)
    num_of_train_sample = TRAIN_RATIO*num_of_total_sample
    num_of_val_sample = VAL_RATIO*num_of_total_sample

    train_set = dataset.sample(n=num_of_train_sample, random_state=42)
    dataset.drop(index=train_set.index, inplace=True)

    val_set = dataset.sample(n=num_of_val_sample, random_state=42)
    dataset.drop(index=val_set.index, inplace=True)

    return Dataset.from_pandas(train_set), Dataset.from_pandas(val_set), Dataset.from_pandas(dataset)

def preprocess_training_examples(examples, tokenizer, max_length, stride):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length= max_length,
        truncation="only_second",
        stride= stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answer"]
    start_positions = []
    end_positions = []
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_test_examples(examples, tokenizer, max_length, stride):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["question"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs
