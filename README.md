# Graph Fusion MBO
The code of this repository is an unofficial extension of [A Graph-Based Approach for Data Fusion andSegmentation of Multimodal Images](https://ieeexplore.ieee.org/document/9206144).

Its goal is semantic segmentation of multimodal images by:
1. Constructing weighted graphs for each input modality
2. Fusing weighted graphs into a similarity matrix W
3. Approximating the l largest eigenvectors of the of the graph laplacian L = I - D^(-1/2)WD^(1/2) through Nyström
4. Running classification on the eigenvectors through an iterative semi-supervised MBO scheme or Spectral Clustering

## Prerequisites
- numpy
- scipy
- sklearn
- matplotlib
- pytorch

## Usage
To use with a different dataset the model expects:
A folder of multimodal numpy images @ ../data-local/images/*dataset*/data
Numpy masks with the same name as their corresponding image @ ../data-local/images/*my_dataset*/mask
Create a new dataset loader file using the custom pytorch loaders @ /dataset/*my_dataset*.py

The model can then be run with default arguments as
```python
python train.py
```
or specified e.g. with 5 percent semi-supervised input and using a different dataset
```python
python train.py --semi-percent 0.05 --dataset s1s2
```

## References
```
@article{9206144,
  author={G. {Iyer} and J. {Chanussot} and A. L. {Bertozzi}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Graph-Based Approach for Data Fusion and Segmentation of Multimodal Images}, 
  year={2020},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2020.2971395}}
```
