
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

model_name = "microsoft/phi-2"
dataset = load_dataset("json", data_files="data/fine_tune_dataset/qa_pairs.jsonl")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05)
model = get_peft_model(model, lora_config)

def tokenize(batch):
    return tokenizer(batch["prompt"], padding="max_length", truncation=True, max_length=256)

tokenized_data = dataset.map(tokenize, batched=True)
training_args = TrainingArguments(
    output_dir="fine_tuned_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"]
)
trainer.train()
