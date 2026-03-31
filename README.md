MIMO-GS
MIMO channel rendering: A 3D-Gaussian Splatting Approach

## Installation
Create the basic environment
```python
conda env create --file environment.yml
conda activate mimogs
```
Install some extensions
```python
cd submodules
pip install ./simple-knn
pip install ./diff-gaussian-rasterization # or cd ./diff-gaussian-rasterization && python setup.py develop
pip install ./fused-ssim
```

Some code snippets are borrowed from [WRF-GS](https://github.com/wenchaozheng/WRF-GS).
