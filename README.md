# Learning from Point Clouds

<details>
<summary>
Installation steps, tested on Ubuntu 18.04. Here's a working procedure for me with <b>Python 3.6 to 3.9</b>.
</summary>

```
conda create --name pygeom python=3.9 -y
conda activate pygeom
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch -y
conda install pyg -c pyg -c conda-forge
conda install ipython -y
pip install matplotlib
pip install class-resolver
pip install wandb
```
</details>

A few general comments:

- If using 3090 GPUs, then those will require CUDA 11.x, change the `cudatoolkit`.
- Installing `pytorch_geometric` with conda can be buggy.
- The `torch-spline-conv` package might be problematic since I have gotten "GLIBC" errors.
  It is an optional dependency of `pytorch_geometric`.
- Despite how the README suggests it requires Python 3.7 or later, I can
  run this with Python 3.6. I can do the above for Python 3.6 to 3.9 and get `import torch` and
  basic stuff like that running. This does _not_ work for Python 3.10 due to incompatibilities
  with `torch`.
- But, [check that `torch` can see the GPU](https://stackoverflow.com/questions/48152674/how-to-check-if-pytorch-is-using-the-gpu).


Careful, Pytorch Geometric is actively getting updated so the example scripts
sometimes use different APIs ([see this change, for example][1]). As of
2022/07/15 I am using version 2.0.4, so if copying any script from the examples
directory, copy the files from [the commit tagged at that version][2] on
2022/03/12.


Try to test:

- Different datasets
- Different hyperparameters (and total number of parameters)
- Different normalization statistics
- And time all of this.

## Supported Comparisons

- PointNet++ and Point Transformer for classification on ModelNet10 and
  ModelNet40. Those datasets come from [this CVPR 2015 paper][3], and papers
  often use ModelNet40 to compare different architectures and algorithms.

- PointNet++ and Point Transformer for segmentation on ShapeNet (but only using
  one class, need to expand on this). Metric seems to be IoU, but is it
  categorical or is it instance level?

For detailed comparisons, please see the `bash/` folder.

[1]:https://github.com/DanielTakeshi/pytorch_geometric/commit/a8601aafd7fc52b87b3f85e86013e64cb7af3e2d
[2]:https://github.com/pyg-team/pytorch_geometric/commit/97d55577f1d0bf33c1bfbe0ef864923ad5cb844d
[3]:https://arxiv.org/abs/1406.5670
