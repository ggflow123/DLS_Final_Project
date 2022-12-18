# DLS_Final_Project

The Final Project of Intro To Deep Learning Class 2022 Fall in NYU

We apply a two stage approach to perform object detection and segmentation.

First, self-supervised learning by MoBY training on Swin Transformer.

Then, use Mask R-CNN with Feature Pyramid Network for object detection.


# Links

This repo contains only self supervised training of Swin Transformer backbone with MoBY.

For DINO on ResNet-50, please consult:

https://github.com/ggflow123/DLS_FINAL_DINO

For Mask R-CNN with FPN, please consult:

https://github.com/ggflow123/DLS_FINAL_mmdetection

# Self-supervised Learning set up:

## There are two algorithms involved: MoBY and DINO.

# MoBY:

Forked from Transformer-SSL github repository. Available at: https://github.com/SwinTransformer/Transformer-SSL

For set up the environment, please check get_started.md in this repository.

I use New York University HPC Greene to complete the running.

In the folder of scripts, there are 4 script files that can submit jobs to the cloud asking for resources. 2 GPUs involved or 4 GPUs invvolved, with the type of RTX8000 or V100. Please put the script files out of the script folder to perform tasks.

For example, to run the script file, just do:

```
sbatch swin_50ep_rtx8000.sbatch
```

To run file with python directly, please do:

```
python -m torch.distributed.launch --nproc_per_node NUM_OF_NODES --master_port 12345  moby_main.py \
--cfg configs/YOUR_CONFIG_FILE --data-path YOUR_DATA_PATH --batch-size 64 [--output YOUR_OUTPUT  --tag YOUR_TAG]
```

configuration files are in the folder of configs.

With 4 V100 GPUs and 92 epochs, we have:

| Epoch | Loss   |
| ----- | ------ |
| 0     | 16     |
| 50    | 8.6240 |
| 60    | 8.8983 |
| 70    | 9.0514 |
| 80    | 8.4456 |
| 90    | 8.9094 |
| 100   | 8.6448 |
| 110   | 8.4562 |
