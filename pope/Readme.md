# 跑POPE数据集

## 本文件夹包含的内容

下载完成的coco文件夹，包含三个ground truth文件
- \coco\coco_pope_random.jsonl - 3000 questions
- \coco\coco_pope_popular.jsonl - 3000 questions
- \coco\coco_pope_adversarial.jsonl - 3000 questions

最好把这三个文件转化为jsonl，但是scripts的问题，不需要管，就用json文件就可以了

# 切换到pope目录
cd /root/autodl-tmp/eval/pope

# 使用镜像源下载coco val2014数据集

wget http://images.cocodataset.org/zips/val2014.zip

或者使用nohup挂到后台并且保存log

nohup wget http://images.cocodataset.org/zips/val2014.zip > download.log 2>&1 &

# 解压
unzip val2014.zip -d /root/autodl-tmp/val2014

# 移动到目标目录并且删除压缩包

mv val2014 /root/autodl-tmp/eval/pope/val2014 && rm val2014.zip

# 使用convert.py将coco文件夹中的3个jsonl文件合并成一个llava_pope_test_origin.jsonl

python convert.py

# 使用全新的eval_pope.py

切换到根目录中
把./llava/eval/eval_pope.py替换成./playground\data\eval\pope\eval_pope.py


cd /
cp /root/autodl-tmp/eval/pope/eval_pope.py ./llava/eval/eval_pope.py