from transformers import AutoModelForQuestionAnswering, AutoTokenizer

def load_model_and_tokenizer(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    return model, tokenizer