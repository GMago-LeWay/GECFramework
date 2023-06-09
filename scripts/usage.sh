# train a gector from pretrained macbert
python main.py --model gector --task_mode train --dataset mucgec --load results/gector-pretrain-20230428-1114 --devices 0,1,2,3 --device 1
# train a gector from pretrained structbert
python main.py --model gector --task_mode train --dataset mucgec --load results/gector-pretrain-20230428-1022 --devices 0,1,2,3 --device 2
# gector inference
python main.py --model gector --task_mode infer --dataset mucgec --load results/gector-mucgec-train-20230523-0215 --devices 0,1,2,3 --device 1

# run pyllama 30B 4bit model