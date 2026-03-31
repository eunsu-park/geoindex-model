import os

BATCH_SIZE = 10
SCRIPT_DIR = "wulver_scripts"

SLURM_HEADER = """#!/bin/bash -l

#SBATCH --job-name={job_name}
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
"""

MODEL = ["convlstm", "transformer", "fusion", "baseline"]
INPUT_DAYS = [list(range(start, 0)) for start in range(-7, 0)]
TARGET_CONFIGS = [
    ([1], 14),
    ([1, 2], 8),
    ([1, 2, 3], 6),
]

# Undersampling mode: "dynamic" (single run per combo) or "static" (k-fold)
UNDERSAMPLING_MODE = "dynamic"


def submit(commands, job_name):
    script = SLURM_HEADER.format(job_name=job_name) + "\n".join(commands)
    with open(f"{SCRIPT_DIR}/{job_name}.sh", "w") as f:
        f.write(script)


def generate_commands():
    for model in MODEL:
        for target_days, num_subsampling in TARGET_CONFIGS:
            for input_days in INPUT_DAYS:
                input_str = f'"sampling.input_days=[{",".join(map(str, input_days))}]"'
                target_str = f'"sampling.target_days=[{",".join(map(str, target_days))}]"'

                if UNDERSAMPLING_MODE == "dynamic":
                    yield ' '.join([
                        'python scripts/train.py',
                        '--config-name=wulver',
                        f'model.model_type={model}',
                        input_str,
                        target_str,
                        'sampling.enable_undersampling=True',
                        'sampling.undersampling_mode=dynamic',
                    ])
                else:
                    for n in range(num_subsampling):
                        yield ' '.join([
                            'python scripts/train.py',
                            '--config-name=wulver',
                            f'model.model_type={model}',
                            input_str,
                            target_str,
                            'sampling.enable_undersampling=True',
                            'sampling.undersampling_mode=static',
                            f'sampling.subsample_index={n}',
                        ])


def main():
    os.makedirs(SCRIPT_DIR, exist_ok=True)
    
    batch = []
    n_command = 0
    n_script = 0

    for command in generate_commands():
        batch.append(command)
        n_command += 1
        print(command)

        if len(batch) == BATCH_SIZE:
            n_script += 1
            submit(batch, f"AP-TRAIN-{n_script:03d}")
            batch = []

    if batch:
        n_script += 1
        submit(batch, f"AP-TRAIN-{n_script:03d}")

    print(f"\nTotal: {n_command} commands, {n_script} scripts")


if __name__ == "__main__":
    main()