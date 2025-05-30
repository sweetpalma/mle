# Machine Learning Experiments

This collection of notebooks documents a series of experiments in machine learning, starting from basic classical machine learning techniques and progressively exploring more complex deep learning approaches.

The primary goal here wasn't necessarily to achieve state of the art results on the first try, but rather to understand the *how* and *why* behind different methodologies. 

This is very much a work in progress. The path is ongoing, new topics will be explored, and the spiral into machine learning will undoubtedly continue. Hopefully, sharing this journey – the frustrations, the breakthroughs, and the relentless pursuit of understanding – might be useful, or at least relatable, to others venturing into this complex and captivating field.

## Running locally

### Preparations

- Install a [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) manager on your PC.
- Prepare a [Kaggle API](https://www.kaggle.com/settings) token and place it here: `~/.kaggle/kaggle.json`

### Installation

- Clone this repository and navigate to it within your shell.
- Setup a Conda environment: `conda env create --yes -f environment.yml`
- Activate it: `conda activate ml`
- Start Jupyter Lab: `jupyter lab`
- Start Jupyter Book: `jupyter book start`

<!-- 
  Update requirements: 
  conda env export | grep -v "^prefix: " > environment.yml 
-->

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

In simple terms, you are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, provided that the original copyright notice and this permission notice are included in all copies or substantial portions of the Software.

## Attribution

If you find the code or explanations in this repository useful for your own work or education, I'd appreciate it if you could provide a reference or link back to this repository. It's not strictly required by the license, but it helps others discover this resource and acknowledges the effort involved. Thank you!
