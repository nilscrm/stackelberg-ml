# Setup
```shell
conda create --file environment.yml
conda activate stackelberg
pip install -e .
```

# Contextualized MAL + PPO
## SimpleMDP
```shell
python stackelberg_mbrl/train_ppo.py
```

# GT-MBRL
## SimpleMDP
```shell
cd _external\mbrl_repo\projects\model_based_npg
python run_model_based_npg.py --output simplemdp --config configs\state_machine.txt
```
