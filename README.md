pii idea is cool :p

## Project structure:
- model stuff goes under `model_src/`
- our actual train function is `train.py`
- data preprocessing utils & synthesis scripts goes under `data_utils/`
- all work and final results should be under `final_notebook.ipynb`

**Note:**
currently our stuff is just unet related stuff, but you baiscally just change the code to fit what we actually want to do. @allenna when you finish you should delete this note

## Setup:
**FIRST, INSTALL MINICONDA!!!**
assumes you have miniconda installed (if not install it, you can test that it works using `conda --version`)
run the following commands in a terminal:
1. `conda env create -f environment_local.yml`
2. make a COPY of the file `.env.example` in the same root, and call it `.env`. please fill in the api keys as necessary
3. download datasets using `get_datasets.sh`


## VERY IMPORTANT!
if you need to update the library with new packages run OR you pulled recnetly and the `environment_local.yml` changed, then: `conda env update -f environment_local.yml --prune`

## Formatter
please download black formatter from vscode extensions

## Training and analysis workflow
- Run one variant: `python scripts/run_variant.py --variant baseline`
- Run all variants: `python scripts/run_all_variants.py`
- Analyze runs: `python scripts/analyze_runs.py --runs baseline o_weight dynamic md`
- Train stacker: `python scripts/fit_meta_stacker.py --base-runs baseline md`
- Evaluate ensembles or stacker: `python scripts/eval_ensemble.py --run baseline` or `python scripts/eval_ensemble.py --stacker outputs/<stacker_run>/stacker.pt`

Note: rerunning the same `--run-name` will overwrite `outputs/<run_name>/` and `models/<run_name>/` fold artifacts.
