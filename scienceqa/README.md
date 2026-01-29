# ScienceQA Dataset

## 说明

本目录包含 ScienceQA 数据集的测试集评估数据，直接 clone 仓库即可使用，无需额外下载。

## 已包含文件

```
scienceqa/
├── images/test/          # 测试集图片（2180张）
├── pid_splits.json       # 数据集划分信息
├── problems.json         # 问题数据（31MB）
├── llava_test_CQM-A.json # LLaVA 测试集格式
└── answers/              # 答案目录
```

## 数据集信息

- **来源**: [ScienceQA](https://github.com/lupantech/ScienceQA)
- **论文**: Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering (NeurIPS 2022)
- **测试集规模**: 2,180 个多模态科学问答题
- **许可证**: CC BY-NC-SA

## 使用方法

直接在代码中引用相对路径即可：

```python
import json

# 加载问题数据
with open('scienceqa/problems.json', 'r') as f:
    problems = json.load(f)

# 加载数据集划分
with open('scienceqa/pid_splits.json', 'r') as f:
    splits = json.load(f)

# 加载 LLaVA 格式的测试数据
with open('scienceqa/llava_test_CQM-A.json', 'r') as f:
    test_data = json.load(f)
```

## 参考链接

- 官方仓库: https://github.com/lupantech/ScienceQA
- 项目主页: https://scienceqa.github.io/
- HuggingFace: https://huggingface.co/datasets/derek-thomas/ScienceQA
