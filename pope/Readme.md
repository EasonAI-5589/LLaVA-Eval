# POPE Evaluation

POPE (Polling-based Object Probing Evaluation) 用于评估视觉语言模型的物体幻觉问题。

## 支持的数据集

| Dataset | Test File | Annotation Dir | Image Folder |
|---------|-----------|----------------|--------------|
| COCO | `coco_test.jsonl` | `pope/coco/` | `pope/val2014/` |
| GQA | `gqa_test.jsonl` | `pope/gqa/` | `gqa/data/images/` |
| AOKVQA | `aokvqa_test.jsonl` | `pope/aokvqa/` | `pope/val2014/` |

每个 annotation 目录包含 3 个文件 (各 3000 条):
- `{dataset}_pope_random.json`
- `{dataset}_pope_popular.json`
- `{dataset}_pope_adversarial.json`

## 目录结构

```
${DATA_DIR}/
├── pope/
│   ├── coco/                          # COCO ground truth
│   │   ├── coco_pope_random.json
│   │   ├── coco_pope_popular.json
│   │   └── coco_pope_adversarial.json
│   ├── gqa/                           # GQA ground truth
│   │   ├── gqa_pope_random.json
│   │   ├── gqa_pope_popular.json
│   │   └── gqa_pope_adversarial.json
│   ├── aokvqa/                        # AOKVQA ground truth
│   │   ├── aokvqa_pope_random.json
│   │   ├── aokvqa_pope_popular.json
│   │   └── aokvqa_pope_adversarial.json
│   ├── coco_test.jsonl                # COCO 测试问题 (9000条)
│   ├── gqa_test.jsonl                 # GQA 测试问题 (9000条)
│   ├── aokvqa_test.jsonl              # AOKVQA 测试问题 (9000条)
│   ├── val2014/                       # COCO val2014 图片 (需下载)
│   └── eval_pope.py                   # 评估脚本
└── gqa/
    └── data/
        └── images/                    # GQA 图片 (需下载)
```

## 下载图片

### COCO val2014 (用于 COCO 和 AOKVQA)
```bash
cd ${DATA_DIR}/pope/
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip
```

### GQA images
从 [GQA 官网](https://cs.stanford.edu/people/dorarad/gqa/download.html) 下载图片，放到 `${DATA_DIR}/gqa/data/images/` 目录。

## 替换评估脚本

用本仓库的 `eval_pope.py` 替换 LLaVA 原本的评估脚本，以支持 COCO、GQA、AOKVQA 三种数据集：

```bash
cp eval_pope.py /path/to/LLaVA/llava/eval/eval_pope.py
```

## 运行脚本 scripts/v1_5/7b/pope.sh

```bash
#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT_DIR="/mnt/bn/bes-mllm-shared/checkpoint/LLaVA"
DATA_DIR="/mnt/bn/bes-mllm-shared/data/LLaVA/LLaVA-Eval"

CKPT="llava-v1.6-vicuna-7b"
SPLIT="coco_test"  # coco_test, gqa_test, aokvqa_test

# 根据数据集设置图片目录和annotation目录
if [ "$SPLIT" == "gqa_test" ]; then
    IMAGE_FOLDER=${DATA_DIR}/gqa/data/images
    ANNO_DIR=${DATA_DIR}/pope/gqa
elif [ "$SPLIT" == "aokvqa_test" ]; then
    IMAGE_FOLDER=${DATA_DIR}/pope/val2014
    ANNO_DIR=${DATA_DIR}/pope/aokvqa
else
    IMAGE_FOLDER=${DATA_DIR}/pope/val2014
    ANNO_DIR=${DATA_DIR}/pope/coco
fi

METHOD=${1}
TOKEN=${2}
PARAM="vtn_${TOKEN}"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path ${CKPT_DIR}/${CKPT} \
        --question-file ${DATA_DIR}/pope/${SPLIT}.jsonl \
        --image-folder ${IMAGE_FOLDER} \
        --answers-file ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks ${CHUNKS} \
        --chunk-idx ${IDX} \
        --pruning_method ${METHOD} \
        --visual_token_num ${TOKEN} \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/merge.jsonl

> "$output_file"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/pope/answers/${SPLIT}/${CKPT}/${METHOD}/${PARAM}/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python llava/eval/eval_pope.py \
    --annotation-dir ${ANNO_DIR} \
    --question-file ${DATA_DIR}/pope/${SPLIT}.jsonl \
    --result-file $output_file
```

## 输出指标

- **Accuracy**: 整体准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 score**: F1 分数
- **Yes ratio**: 回答 "yes" 的比例 (用于检测模型偏向)
