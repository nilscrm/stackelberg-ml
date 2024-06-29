# Setup
```shell
mamba env create -f environment.yml
conda activate stackelberg
pip install -e .
```

# Contextualized MAL + PPO
## SimpleMDP
```shell
python stackelberg_mbrl/train_mal.py
```
or
```shell
python stackelberg_mbrl/train_pal.py
```

# GT-MBRL
```shell
cd _external/mbrl_repo/projects/model_based_npg
conda activate stackelberg
```

## MAL (R)
You may want to change the seed in configs/mal.txt before running
```shell
python run_model_based_npg.py --output simplemdp --config configs/mal.txt
```

## PAL (R)
You may want to change the seed in configs/pal.txt before running
```shell
python run_model_based_npg.py --output simplemdp --config configs/pal.txt
```

## PAL (R) - Convergence
```shell
python run_model_based_npg.py --output simplemdp --config configs/pal.txt
```