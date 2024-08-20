# ViLQA: Vietnamese Legal Question Answering

## Introduction

**ViLQA** is a project focused on fine-tuning Question Answering models for Vietnamese legal text. This project aims to enhance the performance of models in answering questions based on Vietnamese legal documents, particularly on pre-prepared datasets.

## Project Structure

This project implements two approaches to Question Answering:

1. **EQA (Extractive Question Answering)**: This approach extracts the answer directly from the given context.
2. **AG (Answer Generation)**: This approach generates the answer based on the context and question.

The code for each approach is organized in separate directories (`EQA/` and `AG/`) for clarity and maintainability.

```plaintext
ViEQA/
│
├── EQA/
|   ├── config.py                 # Project configuration file
|   ├── train.py                  # Main script for fine-tuning
|   ├── requirements.txt          # List of required Python libraries
|   ├── data/
|   │   ├── __init__.py           # Init module for data
|   │   ├── data_processing.py    # Data processing script
|   │   └── dataset/
|   │       └── ALQAC.csv         # Vietnamese dataset for QA
|   ├── models/
|   │   ├── __init__.py           # Init module for models
|   │   └── eqa_model.py          # Model for Extractive Question Answering
|   └── utils/
|       ├── __init__.py           # Init module for utils
|       └── metrics.py            # Script for calculating model metrics
└── AG/
    ├── ...
    ├── ...
    └── ...
```

### I. What is Extractive Question Answering?

Extractive Question Answering (EQA) is a Natural Language Processing task where the model is given a question and a context (usually a paragraph or a document) and needs to extract the answer to the question directly from the context. The key assumption in EQA is that the answer is a continuous span of text within the given context.

### Problem Formulation

Given:
- A context C = [c1, c2, ..., cn], where ci are tokens in the context
- A question Q = [q1, q2, ..., qm], where qi are tokens in the question

The goal is to find:
- Start index s and end index e in C, so the span [cs, cs+1, ..., ce] answers the question Q.

### Example

Here's an example of Extractive Question Answering:

<figure>
  <p align="center">
    <img src="EQA/images/MRC.png" alt="Fig.1">
  </p>
  <p align="center"><normal>Fig.1: An example of Extractive Question Answering</strong></p>
</figure>


In this example:
- Context: The legal text about the crime of appropriating or destroying relics of fallen soldiers
- Question: "Chiếm đoạt di vật của tử sĩ có thể bị phạt tù lên đến bao nhiêu năm?" (How many years of imprisonment can one face for appropriating relics of fallen soldiers?)
- Answer: "07 năm" (07 years)

The model must extract the correct answer span from the given context based on the question.


## Key Features

- Fine-tuning Extractive Question Answering models on Vietnamese data.
- Support for advanced language models.
- Easy integration with popular tools and libraries.
- Customizable and extendable for different datasets.

## System Requirements

- Python 3.7 or higher
- Python libraries listed in requirements.txt

## Installation Guide

### 1. Clone the repository:

```python
git clone https://github.com/ntphuc149/ViLQA.git
cd ViLQA/EQA
```
### 2. Create a virtual environment (recommended):

```python
python3 -m venv venv
source venv/bin/activate
```
### 3. Install the dependencies:

```python
pip install -r requirements.txt
```

## Usage Instructions
### 1. Configure the project:

Update the parameters in config.py to suit your dataset and requirements.

### 2. Fine-tune and evaluate the model:

Run the following command to start fine-tuning and evaluate the model:

```python
python train.py
```

## Contribution

We welcome contributions to this project. Please create a pull request or open an issue to discuss your ideas for improvement.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
