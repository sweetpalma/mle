# Machine Learning

Here you can observe a bunch of examples I made when learning Machine Learning and Neural Networks.
Feel free to use them for whatever you want.

## Preparations

- Install a [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) manager on your PC.
- Prepare a [Kaggle API](https://www.kaggle.com/settings) token and place it here: `~/.kaggle/kaggle.json`

## Installation

- Clone this repository and navigate to it within your shell.
- Setup a Conda environment: `conda env create --yes -f environment.yml`
- Activate it: `conda activate ml`
- Start Jupyter Lab: `jupyter lab`
- Start Jupyter Book: `jupyter book start`

<!-- 
  Update requirements: 
  conda env export | grep -v "^prefix: " > environment.yml 
-->
