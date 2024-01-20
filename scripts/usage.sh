# train a gector from pretrained macbert
python main.py --model gector --task_mode train --dataset mucgec --load results/gector-pretrain-20230428-1114 --devices 0,1,2,3 --device 1
# train a gector from pretrained structbert
python main.py --model gector --task_mode train --dataset mucgec --load results/gector-pretrain-20230428-1022 --devices 0,1,2,3 --device 2
# gector inference
python main.py --model gector --task_mode infer --dataset mucgec --load results/gector-mucgec-train-20230523-0215 --devices 0,1,2,3 --device 1

# run pyllama 30B 4bit model
#

# train correction glm sft1
python main.py --task_mode train --dataset mucgec --save_root_dir results_glm --load results_glm/correctionglm-pretrain-train-20231120-1615/checkpoint-316000 --devices 0

# get train set inference results
python main.py --task_mode eval_train --dataset mucgec --save_root_dir results_eval --load xxx --devices 0

# train correction glm sft2 (after setting config correctly)
python main.py --task_mode train --dataset mucgec --save_root_dir results_glm --load xxx --devices 0

# infer on correction glm model
python main.py --task_mode infer --dataset mucgec --save_root_dir results_infer --load xxx --devices 0


# custom mode
python main.py --dataset wilocness --task_mode custom --save_root_dir results_main --devices 2 --load results_main/correctionglm-clang8-train_infer-20240114-2117/checkpoint-36000 --config configs/clang8_correctionglm_sft2.json 