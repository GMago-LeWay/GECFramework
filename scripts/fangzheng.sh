# GECToR
python main.py --task_mode infer --save_root_dir results_fangzheng --load results/gector-pretrain-20230428-1114-MARK --model gector --dataset fangzhenggrammar

# DeCoGLM
python main.py --task_mode infer --load results_glm/correctionglm-pretrain-train-20231120-1615/checkpoint-316000 --config configs/pretrain_correctionglm_sft1.json --save_root_dir results_fangzheng --dataset fangzhenggrammar

# DeGLM
## change config detection_only to true
python main.py --task_mode infer --load results_glm/correctionglm-pretrain-train-20231120-1615/checkpoint-316000 --config configs/pretrain_correctionglm_sft1.json --save_root_dir results_fangzheng --dataset fangzhenggrammar
# CoGLM-10B
python main.py --task_mode infer --load results_new/correctionglm-pretrain-train_infer-20240108-1856/checkpoint-74000 --config configs/ch_infer_correctionglm.json --save_root_dir results_fangzheng --dataset fangzhenggrammar

# Baichuan2-13B
## implemented in the llama factory, not here
