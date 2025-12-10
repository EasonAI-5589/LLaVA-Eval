# MME 评测

MME 是一个针对多模态大语言模型的综合评测基准，包含 14 个子任务。

## 数据下载

从 HuggingFace 下载 MME 数据集：
```bash
wget https://huggingface.co/datasets/darkyarding/MME/resolve/main/MME_Benchmark_release_version.zip
unzip MME_Benchmark_release_version.zip
```

## 目录结构

```
${DATA_DIR}/MME/
├── MME_Benchmark_release_version/   # 数据集（14个子任务）
├── eval_tool/                       # 评测工具
│   └── calculation.py
├── answers/                         # 模型输出
├── mme_test.jsonl                   # 标准 MME
├── llava_mme_test.jsonl             # LLaVA 复现版（与标准 MME prompt 不同）
└── convert_answer_to_mme.py
```

## 运行脚本: scripts/v1_5/7b/mme.sh

```bash
#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.5-7b"
SPLIT="llava_mme_test"  # 或 "mme_test"

METHOD=${1}
TOKEN=${2}
PARAM="vtn_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ./playground/data/eval/MME/${SPLIT}.jsonl \
        --image-folder ${DATA_DIR}/MME/MME_Benchmark_release_version \
        --answers-file ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/MME/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

cd ./playground/data/eval/MME

python convert_answer_to_mme.py \
    --data_path ${DATA_DIR}/MME \
    --experiment ${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge

cd eval_tool

python calculation.py --results_dir answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}
```

## 评测步骤

1. 使用 model_vqa_loader 生成模型回答
2. 使用 convert_answer_to_mme.py 转换格式
3. 使用 calculation.py 计算分数

## 输出指标

**感知能力 (Perception)**
- existence, count, position, color, posters, celebrity, scene, landmark, artwork, OCR

**认知能力 (Cognition)**
- commonsense_reasoning, numerical_calculation, text_translation, code_reasoning

## Prompt 差异说明

LLaVA 官方复现版本与标准 MME 的 prompt 格式不同：

**标准 MME (mme_test.jsonl)**:
```
Does this artwork belong to the type of religious? Please answer yes or no.
```

**LLaVA 复现版 (llava_mme_test.jsonl)**:
```
Does this artwork belong to the type of religious?
Answer the question using a single word or phrase.
```

LLaVA 将原始问题中的 `Please answer yes or no.` 替换为换行后追加 `Answer the question using a single word or phrase.`。

注意：`convert_answer_to_mme.py` 会自动将 prompt 转回标准 MME 格式以匹配 GT，因此两种 jsonl 都能正确评测。
