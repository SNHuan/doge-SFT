from transformers import AutoModelForCausalLM, pipeline, AutoTokenizer
import torch

# 加载最新的训练结果
model = AutoModelForCausalLM.from_pretrained(
    "./result/final-result/", 
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16
)

# 确保使用正确的分词器
tokenizer = AutoTokenizer.from_pretrained(
    "./result/final-result/", 
    trust_remote_code=True,
)

# 确保有pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 更稳健的生成参数
generation_config = {
    "max_new_tokens": 512,        # 增加生成长度
    "do_sample": True,
    "temperature": 0.7,           # 调整温度
    "top_p": 0.9,                # 增加采样范围
    "repetition_penalty": 1.1,    # 降低重复惩罚
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "early_stopping": True        # 提前停止生成
}

# 创建文本生成管道
pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer=tokenizer,
    **generation_config
)

# 定义测试用例
test_prompts = [
    "你是谁?",
    "你能做什么?",
    "介绍一下你自己",
    "请写一首关于春天的诗",
    "1+1等于多少?"
]

# 测试生成
print("=" * 50)
print("开始测试模型生成能力")
print("=" * 50)

for prompt in test_prompts:
    print(f"\n输入问题: {prompt}")
    print("-" * 40)
    
    # 确保使用与训练完全一致的格式
    formatted_prompt = f"<s>你是一个名叫Doge的AI助手。请用简洁、友好的方式回答用户的问题。</s>\n<user>{prompt}</user>\n<assistant>"
    
    # 生成回复
    result = pipe(formatted_prompt)[0]["generated_text"]
    
    # 提取assistant的回复部分
    try:
        # 获取assistant标签后的内容
        assistant_content = result.split("<assistant>")[-1].strip()
        # 如果内容中包含</assistant>标签，只取其前面的部分
        if "</assistant>" in assistant_content:
            response = assistant_content.split("</assistant>")[0].strip()
        else:
            response = assistant_content
    except Exception as e:
        print(f"解析回复时出错: {e}")
        response = result
    
    print(f"模型回答: {response}")
    print("=" * 50)

print("\n测试完成!")

# 添加交互式测试
def interactive_test():
    print("\n\n" + "=" * 50)
    print("进入交互模式，输入 'exit' 退出")
    print("=" * 50)
    
    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ['exit', 'quit', '退出']:
            break
            
        # 使用与训练完全一致的格式
        formatted_prompt = f"<s>你是一个名叫Doge的AI助手。请用简洁、友好的方式回答用户的问题。</s>\n<user>{user_input}</user>\n<assistant>"
        
        # 生成回复
        result = pipe(formatted_prompt)[0]["generated_text"]
        
        # 提取assistant的回复部分
        try:
            # 获取assistant标签后的内容
            assistant_content = result.split("<assistant>")[-1].strip()
            # 如果内容中包含</assistant>标签，只取其前面的部分
            if "</assistant>" in assistant_content:
                response = assistant_content.split("</assistant>")[0].strip()
            else:
                response = assistant_content
        except Exception as e:
            print(f"解析回复时出错: {e}")
            response = result
            
        print(f"Doge: {response}")

print("\n是否进入交互模式? (y/n)")
response = input()
if response.lower() in ['y', 'yes', '是']:
    interactive_test()

