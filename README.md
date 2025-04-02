# R-peak Detection

## Commands for training REBAR: 
https://github.com/maxxu05/rebar
- Set up Conda env and packages.
```sh
conda env create -f rebar.yml
conda activate rebar
pip install -e .
```
## Commands for running downstream R-peak detection 
- Data preparation: 
```sh
data/process/ecg_segmentation_processdata.py
```
- Run pipeline:
```sh
python run_exp_downstream.py
```

- Add models in downstream/downstream_nets.py
- Add loss in downstream/downstream_loss.py
- Add models config in downstream/downstream_config.py
- Running pipeline: run_exp_downstream.py



