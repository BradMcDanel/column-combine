# Packing Sparse Convolutional Neural Networks for Efficient Systolic Array Implementations: Column Combining Under Joint Optimization

**README is a work in progress.**

## Abstract
This paper describes a novel approach of packing sparse convolutional neural networks into a denser format for efficient implementations using systolic arrays. By combining multiple sparse columns of a convolutional filter matrix into a single dense column stored in the systolic array, the utilization efficiency of the systolic array can be substantially increased (e.g. 8x) due to the increased density of nonzero weights in the resulting packed filter matrix. In combining columns, for each row, all filter weights but the one with the largest magnitude are pruned. The remaining weights are retrained to preserve high accuracy. We study the effectiveness of this joint optimization for both high utilization efficiency and classification accuracy with ASIC and FPGA designs based on efficient bit-serial implementations of multiplier-accumulators. We demonstrate that in mitigating data privacy concerns the retraining can be accomplished with only fractions of the original dataset (e.g., 10\% for CIFAR-10). We present analysis and empirical evidence on the superior performance of our column combining approach against prior arts under metrics such as energy efficiency (3x) and inference latency (12x).

A lightning talk on the paper can be found [here](https://www.youtube.com/watch?v=9ekKzEKQ1cU).

<p align="center"> 
<img src="https://github.com/BradMcDanel/column-combine/blob/master/figures/column-combine-overview.png" width=700>
</p>

<p align="center"> 
<img src="https://github.com/BradMcDanel/column-combine/blob/master/figures/tile-reduction.png" width=400>
</p>

## Training CNNs with Column Combining
Use `train.py` to train a CNN with column combining. To make it easier to specify the number of layers and number of filters per layer, we parameterize the CNN into commandline arguments. The `--filters` flag takes a list of integers, correspondings to the number of filters in the layer. This is accompanied by the `--layers` flag, which determines how many layers are used for each number of filters. For instance, if the flags are `--filters 128 256 512 --layers 6 6 6`, then the CNN with have a total of 18 convolutional layers (plus one initial layer), with the first 6 layers having 128 filters, then next 6 with 256, and the final 6 with 512.  

Column combining has two main hyperparameters (alpha and gamma) which determine both the amount of weight which are pruned and how efficienctly the sparse matricies can be packed into a denser format. Gamma is set for all layers in a CNN (controlled by the `--gamma` flag). Alpha is set a per layer basis to account for different layers sizes in the CNN. This is set by the `--groups` flag, and follows the same convention as `--filters` and `--layers`. 

### Training a CIFAR-10 Model
A CIFAR-10 model can be trained using the following:
```
python train.py --dataset-root /hdd1/datasets/ --dataset cifar10 --input-size 32 --n-class 10 --aug + --save-path cifar10-cc.pth --batch-size 128 --epochs 300 --lr 0.1 --l1-penalty 1e-7 --gamma 1.75 --filters 128 256 512 --layers 6 6 6 --strides 1 2 2 --groups 1 4 8 --layer-type shift --print-freq 50
```



### Training a ImageNet Model
TODO


