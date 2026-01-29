
https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/pid_splits.json

https://github.com/lupantech/ScienceQA/blob/main/data/scienceqa/problems.json

curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=derek-thomas%2FScienceQA&config=default&split=test&offset=0&length=100"


#!/bin/bash

# 创建目标文件夹
mkdir -p sqa

# 下载 pid_splits.json
echo "Downloading pid_splits.json..."
curl -L -o sqa/pid_splits.json https://github.com/lupantech/ScienceQA/raw/main/data/scienceqa/pid_splits.json

# 下载 problems.json
echo "Downloading problems.json..."
curl -L -o sqa/problems.json https://github.com/lupantech/ScienceQA/raw/main/data/scienceqa/problems.json

# 使用 Hugging Face API 下载测试数据
echo "Downloading test data from Hugging Face..."
curl -X GET \
     "https://datasets-server.huggingface.co/rows?dataset=derek-thomas%2FScienceQA&config=default&split=test&offset=0&length=100" \
     -o sqa/test_data.json

echo "Download complete. Files saved in the 'sqa' folder."


     文件夹名字要改成sqa

     images/test

https://drive.google.com/drive/folders/16kuhXdM-MOhYcFIyRj91WvnDnjnF-xHw

下载test.zip


curl -L -o test.zip "https://drive.google.com/uc?export=download&confirm=t&id=1mAOjwCmRtGpETsVvR2za48IjdnVHPMAx"