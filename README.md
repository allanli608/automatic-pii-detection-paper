pii idea is cool :p

## Project structure:
- model stuff goes under `model_src/`
- our actual train function is `train.py`
- data preprocessing utils & synthesis scripts goes under `data_utils/`
- all work and final results should be under `final_notebook.ipynb`

**Note:**
currently our stuff is just unet related stuff, but you baiscally just change the code to fit what we actually want to do.

## Setup:
assumes you have miniconda installed (if not install it, you can test that it works using `conda --version`)
run the following commands in a terminal:
1. `conda env create -f environment_local.yml`
4. DATA SETUP:
    1. abc
    2. def
5. run `python train.py`


if you need to update the library with new packages run: `conda env update -f environment_local.yml --prune`