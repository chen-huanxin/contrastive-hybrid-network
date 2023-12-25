# Contrastive Hybrid Network

## Usage
``` shell
Python run.py 
    --dataset=DeepTI # or TCIR
    --use_focal  # use focal loss, or use cross-entropy loss
    --gpu=0 # the no. of gpu
    --model=resnet32 # many model to choose, see in params.py
    --mode=cross-entropy # ce-contrastive or performance
```