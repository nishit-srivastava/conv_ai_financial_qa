
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_path = "fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

while True:
    prompt = input("Question: ")
    if not prompt.strip():
        break
    result = gen_pipeline(prompt, max_length=256, do_sample=True)
    print("Answer:", result[0]['generated_text'])
