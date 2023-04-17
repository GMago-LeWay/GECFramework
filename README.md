# Chinese GEC(Grammatical Error Correction)任务代码系统

## 功能描述

系统主要提供三种功能：

- GEC任务模型的训练
- GEC结果的评估（未完全开发）
- GEC数据增广程序入口

## 程序入口

main.py的args参数说明：

--model 使用的语法纠错模型，现支持bert/softmaskedbert.

--task_mode 执行的任务，现支持：

- train 在指定的数据集上训练出新的GEC模型并在测试集上导出结果保存（随机种子可在main.py指定）
- tune 在指定的数据集上训练GEC模型，进行100次调参（次数在main.py指定）
- test 在有标注的数据集上测试load出模型的GEC效果，结果保存在指定目录下，包含json文件和\t间隔的txt文件，顺序为原始文本 /t 正确文本 /t 预测文本
- infer 在无标注的数据上进行load出的GEC模型的推断，结果保存在指定目录下，包含json文件，每个条目只有原始文本和预测文本

--dataset 使用的数据集，现支持：

- hybridset/fangzhengtest/peopledaily/augment
- 数据集需要在config.py里指定目录

--model_save_dir 模型存储的根目录，会按log_order指定的数字生成子目录train_{log_order}存储训练过程中产生的中间checkpoint

--res_save_dir 结果存储的目录，实验结果、测试与推断结果都会存储在这里

--device 使用的GPU的序号

--load 加载模型checkpoint的路径，加载的文件对应的模型要与--model指定的模型配置一致

- TODO: 加载优化器

--log_order 存储日志的序号和存储模型checkpoint的子目录序号，防止相同模型相同数据训练时冲突

- 日志存储于log目录下，按照任务-模型-数据集-order命名

--csv_order 存储实验效果的表格文件的序号

--data_save_dir 执行数据增广任务时，数据的存储路径

## 目录和文件功能说明

main.py 使用入口

config.py 模型以及数据的配置

data.py 数据加载的方法

augmentation.py 数据增广的方法

metrics.py 评价指标的计算

models/ GEC模型的定义

trainers/  GEC模型训练方法的定义

results/ 模型存储以及实验结果存储

utils/ 工具脚本

fangzheng_v2/ 方正相关

# 使用实例

使用BERT在augment数据集上进行语法纠错模型的训练

python main.py --model bert --task_mode train --dataset augment --device 0

使用训练好的BERT模型在方正测试集上进行模型效果的测试并生成结果

python main.py --model bert --task_mode test --dataset fangzhengtest --load results/models/train_0/bert-augment.pth --device 2

结果在results/results下生成。
