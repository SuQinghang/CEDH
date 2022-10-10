# Efficient Hash Code Expansion by Recycling Old Bits

## Requirements
This repository hash been built and tested around `Python3.8` and `PyTorch1.10`.

In addition, we use the `loguru` library to record the training process. One can install it with 
`pip install loguru`

## Quick Start
### Use ADSH to generate database hash codes with k bits

```
python run.py --method adsh --dataset cifar-10 --root <path of cifar-10> --code-length k
```

### Use CEDH to expand k bits database hash codes to k' bits
```
python run.py --method cedh --dataset cifar-10 --root <path of cifar-10> \
--code-length k' --original-method adsh --original-dir <path of adsh checkpoint> --original-length k
```

More detailed explaination of the arguments can be fund in `run.py/load_config()`
## Results

### No new data
1. We train ADSH with different number of bits from scratch, and use learned CNN models to generate hash codes of the whole database and query data.
2. As for CEDH, we learned CNN models and projection matrices based on ADSH(20,28,44,60bits). Then we use projection matrices to expand all database hash codes to longer length(24,32,48,64bits),
and CNN models are used for generating hash codes of query images.

|length(bits)|20|24|28|32|44|48|60|64|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|ADSH|0.7538|0.7572|0.7614|0.7681|0.7723|0.7748|0.7739|0.7760|
|CEDH||0.7583||0.7639||0.7789||0.7795|

### New data 
Here, we simulate a condition that new images (but no new classes) arrive. Specifically, we split the database images into two parts. 4/5 are considered as existing database,
    and the other 1/5 are newly emerging images.
1. For ADSH, either use the existing CNN model that generates k bits to generate hash codes of new images
    , or retrain a CNN model that generates k+c bits and regenerate the hash codes of existing database images and new images.
2. For CEDH, we only need to use the projection matrix to expand the existing database codes and use the CNN model to generate the hash codes of new images.

|length(bits)|20|24|28|32|44|48|60|64|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|ADSH|0.7538|0.7572|0.7614|0.7681|0.7723|0.7748|0.7739|0.7760|
|CEDH|0.7658|0.7660|0.7699|0.7708|0.7869|0.7877|0.7833|0.7848|

## Acknowledgement
The implementation of `ADSH` was modified from https://github.com/TreezzZ/ADSH_PyTorch