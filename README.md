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
## SimpleMDP
```shell
cd _external/mbrl_repo/projects/model_based_npg
python run_model_based_npg.py --output simplemdp --config configs/state_machine.txt
```
