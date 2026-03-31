#!/bin/bash -l

#SBATCH --job-name=AP-TRAIN-065
#SBATCH --output=/home/hl545/ap/final/train_outs/%x.%j.out
#SBATCH --error=/home/hl545/ap/final/train_errs/%x.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=8000M
#SBATCH --qos=high_wangj
#SBATCH --account=wangj
#SBATCH --time=7-00:00:00

module purge > /dev/null 2>&1
module load wulver
conda activate ap
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-4,-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=10
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-4,-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=11
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-4,-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=12
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-4,-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=13
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=0
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=1
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=2
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=3
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=4
python scripts/train.py --config-name=wulver model.model_type=baseline "sampling.input_days=[-3,-2,-1]" "sampling.target_days=[1]" sampling.enable_undersampling=True sampling.subsample_index=5