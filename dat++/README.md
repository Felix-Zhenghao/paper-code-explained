Original code: https://github.com/LeapLabTHU/DAT


For installation (tested on cuda 11.7):

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
export CUDA_HOME=/usr/local/cuda-11.7
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
```

If cuda 11.7 is not available, maybe just install a pytorch version according to the local cuda version [here](https://pytorch.org/get-started/previous-versions/).
