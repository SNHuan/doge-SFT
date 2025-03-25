import torch
from datasets import load_dataset
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from utils.Util import Util
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 设置是否使用fp16训练 - 使用FP16可能会提高训练稳定性
USE_FP16 = True

# 使用新的API方式，直接传递所需参数
dataset = Util.load_or_download_dataset(
    "Congliu/Chinese-DeepSeek-R1-Distill-data-110k-SFT", 
    split="train[:20000]"  # 增加数据量
)
print(dataset)

# 使用新的API方式，直接传递所有参数
# 确保使用一致的分词器路径
tokenizer = Util.load_or_download_tokenizer(
    "Doge-tokenizer", 
    max_length=512,
    padding_side="right"  # 确保填充在右侧
)

# 确保tokenizer有特殊token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 改进的聊天格式，更清晰地分隔输入和输出
def format_prompt(prompt):
    return {
        "text": f"<s>你是一个名叫Doge的AI助手。请用简洁、友好的方式回答用户的问题。</s>\n<user>{prompt['instruction']}</user>\n<assistant>{prompt['output']}</assistant>"
    }

dataset = dataset.map(format_prompt, remove_columns=dataset.column_names)
print(dataset)

# 使用新的API方式加载模型，直接传递所有参数
model = Util.load_or_download_model(
    "SmallDoge-60M",
    device_map="auto",
    torch_dtype=torch.float16
)

# 确保模型处于训练模式
model.train()

# 准备模型进行LoRA微调
model = prepare_model_for_kbit_training(model)

# 调整LoRA配置 - 增加目标层和参数
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 扩展目标层
    modules_to_save=["lm_head"]
)

# 应用LoRA配置
model = get_peft_model(model, peft_config)

# 打印可训练参数和冻结参数
trainable_params = 0
all_param = 0
for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
print(
    f"可训练参数: {trainable_params} ({100 * trainable_params / all_param:.2f}% 的总参数)"
)

# 确保模型参数正确设置为可训练状态
for name, param in model.named_parameters():
    if "lora" in name:  # 确保所有LoRA参数可训练
        param.requires_grad = True

trainer_arg = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=4,     # 增加批次大小
    gradient_accumulation_steps=4,      # 减少梯度累积步数
    optim="adamw_torch",
    learning_rate=2e-5,                # 略微提高学习率
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,                # 增加训练轮次
    logging_steps=10,
    bf16=True,
    gradient_checkpointing=True,
    save_strategy="epoch",             # 改为每个epoch保存一次
    # 下面这些设置有助于稳定训练
    dataloader_drop_last=True,
    ddp_find_unused_parameters=False,
    no_cuda=False,
    remove_unused_columns=True,
    # 梯度裁剪避免梯度爆炸
    max_grad_norm=1.0,
    # 添加评估
    eval_strategy="epoch",             # 改为每个epoch评估一次
    # 添加checkpoint
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# 创建训练集和验证集
train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

# 使用旧版TRL的SFTTrainer，只保留支持的参数
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=trainer_arg,
    processing_class=tokenizer    # 旧版TRL使用processing_class参数
)

print("开始训练...")
print(f"模型类型: {type(model)}")
print(f"是否有可训练参数: {any(p.requires_grad for p in model.parameters())}")


print("前向测试以验证模型是否能接收输入...")
test_input = tokenizer("测试输入", return_tensors="pt").to(model.device)
with torch.no_grad():
    test_output = model(**test_input)
print("前向测试完成，模型可以输出。")

# 创建保存目录
os.makedirs("./result/final-result", exist_ok=True)

try:
    trainer.train()
    # 保存模型和分词器
    trainer.model.save_pretrained("./result/final-result")
    tokenizer.save_pretrained("./result/final-result")
    print("训练完成，模型已保存!")
except Exception as e:
    print(f"训练过程中发生错误: {e}")
    # 尝试保存部分训练的模型
    print("尝试保存部分训练的模型...")
    trainer.model.save_pretrained("./result/partial-result")
    tokenizer.save_pretrained("./result/partial-result")
    raise