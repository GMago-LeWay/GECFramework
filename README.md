# Chinese GEC(Grammatical Error Correction)任务代码系统

## 功能描述

系统主要提供三种功能：

- 各种GEC任务模型的训练、推断
- GEC数据增广程序入口（待开发）


##  环境配置
python>=3.9  
1. 先创建新的虚拟环境并安装符合你环境的pytorch

    `pip install -r requirements.txt`

2. 改动数据集和模型基础目录：config.py中的  
- DATA_ROOT_DIR = '/data/liwei/datasets'  
- MODEL_ROOT_DIR = '/data/liwei/models'  

3. 下载并处理数据集，确定数据集的相对路径（config.py中的DATA_DIR_NAME）
- 例：main.py中启动程序时，若指定--dataset fangzhenggrammar，则会根据`DATA_ROOT_DIR + DATA_DIR_NAME['fangzhenggrammar']`确定获取数据的目录

4. 下载一些要求的huggingface模型到MODEL_ROOT_DIR
- 例如GECToR模型需要使用chinese-macbert-base的权重

5. 后处理相关，不需要则可直接跳过
- 如果需要使用后处理（目前仅限correctionglm, seq2seqbeta, seq2span）
    - 为了进行指定的后处理，还需要安装spacy并下载语言包  
        - `python -m spacy download en_core_web_sm`
    - 如果希望使用基于ERRANT的评测（CoNLL-14数据集上需要），需要安装python2，可以直接用conda创建：  
        - `conda create --name m2 python=2.7`
        - 改动postprocess中的PYTHON2_PATH为你安装python2的路径  
        - (激活m2环境，使用pip安装nltk：如果用指定的m2评测文件则不需要)
    - 后处理中的目录同样需要改动：postprocess.py中的
        - CONLL14_M2_FILE = 'utils/m2scorer/official-2014.combined.m2'  
        - BEA_DEV_M2_FILE = '../datasets/BEA19_dev/ABCN.dev.gold.bea19.m2'  
        - MUCGEC_DEV_M2_FILE = '../datasets/MuCGEC/MuCGEC_dev/valid.gold.m2.char'  
        - FCGEC_DEV_FILE = '../datasets/FCGEC/FCGEC_dev/test.json'  

请更改到你的路径

## 程序入口

main.py的args参数说明：

`--model` 使用的语法纠错模型，现支持bert/softmaskedbert/stgjoint/seq2seq/seq2edit/gector/llm/chinese_llama/llama/llama_quant/chatglm/correctionglm/seq2seqbeta/seq2span.
- 主要推荐使用的有bert/softmaskedbert/stgjoint/gector/correctionglm/seq2seqbeta/seq2span，基本上既支持训练也支持推断
- 大模型的纠错我们打算在后续新建另一个repo

`--task_mode` 执行的任务，大多数支持：
- train 在指定的数据集上训练出新的GEC模型并在测试集上导出结果保存（随机种子可在main.py指定）
- infer 在无标注的数据上进行load出的GEC模型的推断，结果保存在指定目录下，包含json文件，每个条目只有原始文本和预测文本
- 少部分模型支持一些特殊的模式

`--dataset` 使用的数据集，根据下载数据的情况，可以在config.py中自行扩展：

- 目前使用到的数据集包括hybridset/nlpcc2018task2/fangzhengspell/fangzhenggrammar/guangming/peopledaily/augment/fangzhengaugment/fangzhengdapei/fcgec/mucgec/pretrain/c4/lang8/clang8/fce/nucle/wilocness/hybrid等，但需要自行下载并处理
- 数据集需要在config.py里指定目录

`--save_root_dir` 结果存储的根目录，每次运行程序会产生一个模型-数据集-对应时间的目录用于存储结果，在该根目录下新建

`--devices` 使用的多卡环境，相当于CUDA_VISIBLE_DEVICES（某些可多卡训练的模型需要指定，e.g. seq2seq）

`--device` 使用的GPU的序号

`--load` 加载模型checkpoint的路径，一般加载的是训练产生结果的目录

`--seed` 随机数种子

`--data_save_dir` 执行数据增广任务时，数据的存储路径

`--config` 使用--model=correctionglm，即DeCoGLM系列模型时需要指定具体的配置文件


## 使用实例

使用GECToR在pretrain数据集上进行语法纠错模型的训练

`python main.py --task_mode train --save_root_dir results --model gector --dataset pretrain`

使用训练好的GECToR模型在方正测试集上进行模型效果的测试并生成结果

`python main.py --task_mode infer --save_root_dir results_fangzheng --load results/gector-pretrain-20230428-1114-MARK --model gector --dataset fangzhenggrammar`

模型的配置请到config.py中Config类中的对应部分调整（DeCoGLM除外）。Config.MODEL_MAP字典中存储着各个模型的配置函数

## 关于DeCoGLM的用法
DeCoGLM(--model correctionglm)的配置不位于config.py中
scripts/usage.sh

## 程序大致原理
main.py中，主函数会接收各种argument然后根据模型选择Experiments，注意
- bert/softmaskedbert/stgjoint/gector使用的是稍旧一些的系统
- correctionglm/seq2seqbeta/seq2span使用的是新的系统，其特征是采用huggingface的transformers和datasets实现数据上的统一

除了DeCoGLM可以用--config来指定配置文件以外，其他模型的配置需要到config.py中Config类中的对应部分调整（DeCoGLM除外）。Config.MODEL_MAP字典中存储着各个模型的配置函数

不同的--task_mode将会导向Experiment中不同的run函数，例如训练和推理的函数就有所不同，但都大致包含以下过程：
- 加载配置config
- 加载数据集（或dataloader）
- 初始化模型
- 加载模型（如有--load）
- 开始训练/推断
- 结果的输出或保存

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
