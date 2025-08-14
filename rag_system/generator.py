
from transformers import pipeline

def get_generator(model_name="microsoft/phi-3-mini-4k-instruct"):
    return pipeline("text-generation", model=model_name)

def generate_answer(generator, context, question):
    prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
    return generator(prompt, max_length=256, do_sample=True)[0]['generated_text']
