# GECFramework (Grammatical Error Correction Framework)

| [English](README.md) | [中文](README_cn.md) |

## Functional Description

The system primarily offers three functionalities:

- Training and inference of various GEC task models
- Entry point for GEC data augmentation program (to be developed)

## Environment Configuration
Python >= 3.9

1. First, create a new virtual environment and install PyTorch that matches your environment.

`pip install -r requirements.txt`


2. Modify the base directories for datasets and models in `config.py`:
- `DATA_ROOT_DIR = '/data/liwei/datasets'`
- `MODEL_ROOT_DIR = '/data/liwei/models'`

3. Download and process the datasets to determine the relative path of the datasets (`DATA_DIR_NAME` in `config.py`).
- For example: When starting the program in `main.py`, if you specify `--dataset fangzhenggrammar`, the directory to obtain the data will be determined by `DATA_ROOT_DIR + DATA_DIR_NAME['fangzhenggrammar']`.

4. Download some required Hugging Face models to `MODEL_ROOT_DIR`.
- For instance, the GECToR model requires the weights of `chinese-macbert-base`.

5. Post-processing related, skip if not needed.
- If you need to use post-processing (currently limited to correctionglm, seq2seqbeta, seq2span):
  - To perform the specified post-processing, you also need to install Spacy and download the language package.
    ```
    python -m spacy download en_core_web_sm
    ```
  - If you wish to use ERRANT-based evaluation (required for CoNLL-14 dataset), you need to install Python 2, which can be done directly with Conda:
    ```
    conda create --name m2 python=2.7
    ```
  - Modify `PYTHON2_PATH` in `postprocess` to the path where you installed Python 2.
  - (Activate the m2 environment and install `nltk` with pip: not needed if you are using a specified m2 evaluation file)
  - The directories in post-processing also need to be changed in `postprocess.py`:
    - `CONLL14_M2_FILE = 'utils/m2scorer/official-2014.combined.m2'`
    - `BEA_DEV_M2_FILE = '../datasets/BEA19_dev/ABCN.dev.gold.bea19.m2'`
    - `MUCGEC_DEV_M2_FILE = '../datasets/MuCGEC/MuCGEC_dev/valid.gold.m2.char'`
    - `FCGEC_DEV_FILE = '../datasets/FCGEC/FCGEC_dev/test.json'`

Please change to your path.

## Program Entry

Explanation of args parameters in `main.py`:

`--model` The syntax correction model used, currently supports bert/softmaskedbert/stgjoint/seq2seq/seq2edit/gector/llm/chinese_llama/llama/llama_quant/chatglm/correctionglm/seq2seqbeta/seq2span.
- The main recommended ones are bert/softmaskedbert/stgjoint/gector/correctionglm/seq2seqbeta/seq2span, which basically support both training and inference.
- We plan to build a separate repo for large model error correction later.

`--task_mode` The task to be executed, most support:
- `train` Train a new GEC model on the specified dataset and save the results on the test set (random seed can be specified in `main.py`).
- `infer` Infer the loaded GEC model on unlabeled data, and save the results in the specified directory, including a JSON file with each entry containing only the original text and the predicted text.
- A few models support some special modes.

`--dataset` The dataset used, expandable in `config.py` based on the downloaded data:

- The datasets currently used include hybridset/nlpcc2018task2/fangzhengspell/fangzhenggrammar/guangming/peopledaily/augment/fangzhengaugment/fangzhengdapei/fcgec/mucgec/pretrain/c4/lang8/clang8/fce/nucle/wilocness/hybrid, etc., but need to be downloaded and processed by yourself.
- Datasets need to be specified in the directory in `config.py`.

`--save_root_dir` The root directory for storing results. Each time the program runs, it will create a directory for the model-dataset-corresponding time to store the results under this root directory.

`--devices` The multi-card environment used, equivalent to CUDA_VISIBLE_DEVICES (some models that can be trained with multiple cards need to be specified, e.g., seq2seq).

`--device` The number of the GPU used.

`--load` The path to load the model checkpoint, which is usually the directory where the training results are generated.

`--seed` Random number seed.

`--data_save_dir` The storage path for data when performing data augmentation tasks.

`--config` When using `--model=correctionglm`, i.e., the DeCoGLM series of models, you need to specify a specific configuration file.

## Usage Examples

Train the GECToR model for syntax correction on the pretrain dataset:

`python main.py --task_mode train --save_root_dir results --model gector --dataset pretrain`

Infer using the trained GECToR model on the mucgec test set:

`python main.py --task_mode infer --save_root_dir results --load results/gector-pretrain-20230428-1114-MARK --model gector --dataset mucgec`


Adjust the model configuration in the corresponding section of the Config class in `config.py` (except for DeCoGLM). The Config.MODEL_MAP dictionary stores the configuration functions of each model.

## About Using DeCoGLM
The configuration for DeCoGLM (--model correctionglm) is not located in `config.py`.
**For using DeCoGLM from the paper, please refer to `scripts/usage.sh`**. The README will be updated later, and the current repository is the first version.

## Program General Principle
In `main.py`, the main function will receive various arguments and then select Experiments based on the model. Note:
- bert/softmaskedbert/stgjoint/gector use a slightly older system.
- correctionglm/seq2seqbeta/seq2span use a new system, characterized by the use of Hugging Face's transformers and datasets to achieve data unification.

Except for DeCoGLM, which can specify a configuration file with `--config`, the configuration for other models needs to be adjusted in the corresponding section of the Config class in `config.py` (except for DeCoGLM). The Config.MODEL_MAP dictionary stores the configuration functions of each model.

Different `--task_mode` will lead to different run functions in Experiments. For example, the functions for training and inference are different, but they generally include the following processes:
- Load configuration `config`
- Load the dataset (or dataloader)
- Initialize the model
- Load the model (if there is --load)
- Start training/inference
- Output or save the results

## Directory and File Function Description

`main.py` - Entry point for use

`config.py` - Configuration for models and data

`data.py` - Method for loading data

`augmentation.py` - Method for data augmentation

`metrics.py` - Calculation of evaluation metrics

`models/` - Definition of GEC models

`trainers/` - Definition of GEC model training methods

`results/` - Storage of model and experimental results

`utils/` - Utility scripts

