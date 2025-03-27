import datasets
import torch
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config
)

model_file = './models/model/SmallDoge-60M/'
tokenizer_file = './models/tokenizer/Doge-tokenizer/'
datset_file = './models/dataset/Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT/'

model = AutoModelForCausalLM.from_pretrained(
    model_file,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(tokenizer_file,trust_remote_code=True)

dataset = load_from_disk(datset_file)

def prompt_func(prompt):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt["instruction"]},
        {"role": "assistant", "content": prompt["output"]}
    ]
    
    return {"text" : tokenizer.apply_chat_template(message,tokenize=False)}

dataset = dataset.map(prompt_func, remove_columns=dataset.column_names)

print(dataset[0]["text"])

peft_config = LoraConfig(
    r = 8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

train_config = SFTConfig(
    gradient_accumulation_steps=1,
    learning_rate=2e-4,
    weight_decay=0.1,
    num_train_epochs=1,
    save_strategy="steps",
    save_steps=100,
    max_steps=500,
    optim="adamw_torch",
    bf16=True,
    gradient_checkpointing=True,
    output_dir="./output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_steps=10,
    
    dataset_text_field="text"
)

trainer = SFTTrainer(
    args=train_config,
    train_dataset=dataset,
    model=model,
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()

trainer.save_model("./result/final-result/")