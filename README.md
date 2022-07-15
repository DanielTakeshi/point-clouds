# point-clouds

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

<
