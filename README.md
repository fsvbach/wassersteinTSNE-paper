# Wasserstein t-SNE 
###### Fynn Bachmann, Philipp Hennig & Dmitry Kobak

This repository reproduces the figures in the _Wasserstein t-SNE_ paper in the proceedings of [ECML/PKDD 2022](https://link.springer.com/chapter/10.1007/978-3-031-26387-3_7).

For the python package `WassersteinTSNE` see the repository [WassersteinTSNE](https://github.com/fsvbach/WassersteinTSNE). 

## Dependencies 

This repository uses the WasserteinTSNE package *version 1.1.0* and its dependencies

- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `openTSNE`
- `scikit-learn`
- `igraph`
- `leidenalg`

## Usage

Running `python main.py` will reproduce all Figures in `Figures/Figure{i}.pdf`. This takes about two minutes when using the cached files (default). To recompute these files please see the respective lines in `main.py` or `Experiments/Figure{i}.py`.

## Data

The dataset used in the paper is publically available at the website of the [Bundeswahlleiter](https://tinyurl.com/mpevp355). It is also included in the repository at `Datasets/GER2017/...`. 

A second dataset, the European Values Study (which wasn't used in the paper) is also available at [GESIS](https://doi.org/10.4232/1.13560) and in `Datasets/EVS2020/...`.

## Figures

All figures are already included in this repository at `Figures/Figures{i}.pdf`. See `params.py` for customization.

