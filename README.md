# Doge SFT

基于小型模型的LoRA SFT训练项目，用于训练一个名为"Doge"的中文AI助手。

## 项目结构

- `run.py`: 主训练脚本
- `test.py`: 模型测试脚本
- `utils/`: 辅助工具模块

## 训练方法

本项目使用LoRA微调方法，针对约20,000条中文指令数据进行训练，使模型学习成为一个友好的AI助手。

## 使用方法

1. 安装依赖
```
pip install torch transformers datasets peft trl
```

2. 运行训练
```
python run.py
```

3. 测试模型
```
python test.py
```

## 特点

- 使用LoRA技术进行高效微调
- 自定义指令格式
- 简单易用的测试接口 