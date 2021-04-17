# Graph Fusion MBO
The code of this repository is an unofficial extension of [A Graph-Based Approach for Data Fusion andSegmentation of Multimodal Images](https://ieeexplore.ieee.org/document/9206144).

Its goal is semantic segmentation of multimodal images by:
1. Constructing weighted graphs for each input modality
2. Fusing weighted graphs into a similarity matrix W
3. Approximating the l largest eigenvectors of the of the graph laplacian L = I - D^(-1/2)WD^(1/2) through Nyström
4. Running classification on the eigenvectors through an iterative semi-supervised MBO scheme or Spectral Clustering

## Prerequisites


## References
'''
@article{9206144,
  author={G. {Iyer} and J. {Chanussot} and A. L. {Bertozzi}},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={A Graph-Based Approach for Data Fusion and Segmentation of Multimodal Images}, 
  year={2020},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TGRS.2020.2971395}}
'''
