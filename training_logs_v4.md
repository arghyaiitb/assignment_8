
================================================================================
# Training Session Started: 2025-10-21 03:52:45
================================================================================

## Critical Changes Applied (Based on Analysis & ref_model)
✅ **MaxPool Removed**: Preserves 32x32 spatial resolution through layer1
✅ **Dropout Disabled**: Set to 0.0 in residual blocks (was 0.15)
✅ **CutMix Augmentation**: Enabled with prob=0.3, alpha=1.0
✅ **Enhanced Augmentation**: Added PadIfNeeded+RandomCrop, GaussNoise
✅ **Deeper Architecture**: Changed to [5,5,5,5] blocks (20 total, was 16)
✅ **Increased FC Dropout**: 0.3 (was 0.25)
✅ **Optimized LR Schedule**: max_lr=0.1, div_factor=25, final_div_factor=1000
   └─ Fixes volatility issue, enables better convergence
Expected Impact: +15-22% test accuracy improvement (TARGET: 75%+)

CUDA Available? True
cuDNN benchmark enabled for maximum speed

## Model Configuration
- Architecture: ResNet34-Enhanced (Modified for CIFAR-100)
- Block Structure: [5, 5, 5, 5] = 20 residual blocks (was [3,4,6,3]=16)
- MaxPool: REMOVED (preserves 32x32 spatial resolution)
- Dropout in residual blocks: 0.0 (disabled)
- FC Layer Dropout: 0.3 (increased from 0.25)
- Number of classes: 100

Using channels_last memory format for optimal T4 performance
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13           [-1, 64, 32, 32]          36,864
      BatchNorm2d-14           [-1, 64, 32, 32]             128
           Conv2d-15           [-1, 64, 32, 32]          36,864
      BatchNorm2d-16           [-1, 64, 32, 32]             128
       BasicBlock-17           [-1, 64, 32, 32]               0
           Conv2d-18           [-1, 64, 32, 32]          36,864
      BatchNorm2d-19           [-1, 64, 32, 32]             128
           Conv2d-20           [-1, 64, 32, 32]          36,864
      BatchNorm2d-21           [-1, 64, 32, 32]             128
       BasicBlock-22           [-1, 64, 32, 32]               0
           Conv2d-23           [-1, 64, 32, 32]          36,864
      BatchNorm2d-24           [-1, 64, 32, 32]             128
           Conv2d-25           [-1, 64, 32, 32]          36,864
      BatchNorm2d-26           [-1, 64, 32, 32]             128
       BasicBlock-27           [-1, 64, 32, 32]               0
           Conv2d-28          [-1, 128, 16, 16]          73,728
      BatchNorm2d-29          [-1, 128, 16, 16]             256
           Conv2d-30          [-1, 128, 16, 16]         147,456
      BatchNorm2d-31          [-1, 128, 16, 16]             256
           Conv2d-32          [-1, 128, 16, 16]           8,192
      BatchNorm2d-33          [-1, 128, 16, 16]             256
       BasicBlock-34          [-1, 128, 16, 16]               0
           Conv2d-35          [-1, 128, 16, 16]         147,456
      BatchNorm2d-36          [-1, 128, 16, 16]             256
           Conv2d-37          [-1, 128, 16, 16]         147,456
      BatchNorm2d-38          [-1, 128, 16, 16]             256
       BasicBlock-39          [-1, 128, 16, 16]               0
           Conv2d-40          [-1, 128, 16, 16]         147,456
      BatchNorm2d-41          [-1, 128, 16, 16]             256
           Conv2d-42          [-1, 128, 16, 16]         147,456
      BatchNorm2d-43          [-1, 128, 16, 16]             256
       BasicBlock-44          [-1, 128, 16, 16]               0
           Conv2d-45          [-1, 128, 16, 16]         147,456
      BatchNorm2d-46          [-1, 128, 16, 16]             256
           Conv2d-47          [-1, 128, 16, 16]         147,456
      BatchNorm2d-48          [-1, 128, 16, 16]             256
       BasicBlock-49          [-1, 128, 16, 16]               0
           Conv2d-50          [-1, 128, 16, 16]         147,456
      BatchNorm2d-51          [-1, 128, 16, 16]             256
           Conv2d-52          [-1, 128, 16, 16]         147,456
      BatchNorm2d-53          [-1, 128, 16, 16]             256
       BasicBlock-54          [-1, 128, 16, 16]               0
           Conv2d-55            [-1, 256, 8, 8]         294,912
      BatchNorm2d-56            [-1, 256, 8, 8]             512
           Conv2d-57            [-1, 256, 8, 8]         589,824
      BatchNorm2d-58            [-1, 256, 8, 8]             512
           Conv2d-59            [-1, 256, 8, 8]          32,768
      BatchNorm2d-60            [-1, 256, 8, 8]             512
       BasicBlock-61            [-1, 256, 8, 8]               0
           Conv2d-62            [-1, 256, 8, 8]         589,824
      BatchNorm2d-63            [-1, 256, 8, 8]             512
           Conv2d-64            [-1, 256, 8, 8]         589,824
      BatchNorm2d-65            [-1, 256, 8, 8]             512
       BasicBlock-66            [-1, 256, 8, 8]               0
           Conv2d-67            [-1, 256, 8, 8]         589,824
      BatchNorm2d-68            [-1, 256, 8, 8]             512
           Conv2d-69            [-1, 256, 8, 8]         589,824
      BatchNorm2d-70            [-1, 256, 8, 8]             512
       BasicBlock-71            [-1, 256, 8, 8]               0
           Conv2d-72            [-1, 256, 8, 8]         589,824
      BatchNorm2d-73            [-1, 256, 8, 8]             512
           Conv2d-74            [-1, 256, 8, 8]         589,824
      BatchNorm2d-75            [-1, 256, 8, 8]             512
       BasicBlock-76            [-1, 256, 8, 8]               0
           Conv2d-77            [-1, 256, 8, 8]         589,824
      BatchNorm2d-78            [-1, 256, 8, 8]             512
           Conv2d-79            [-1, 256, 8, 8]         589,824
      BatchNorm2d-80            [-1, 256, 8, 8]             512
       BasicBlock-81            [-1, 256, 8, 8]               0
           Conv2d-82            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-83            [-1, 512, 4, 4]           1,024
           Conv2d-84            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-85            [-1, 512, 4, 4]           1,024
           Conv2d-86            [-1, 512, 4, 4]         131,072
      BatchNorm2d-87            [-1, 512, 4, 4]           1,024
       BasicBlock-88            [-1, 512, 4, 4]               0
           Conv2d-89            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-90            [-1, 512, 4, 4]           1,024
           Conv2d-91            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-92            [-1, 512, 4, 4]           1,024
       BasicBlock-93            [-1, 512, 4, 4]               0
           Conv2d-94            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-95            [-1, 512, 4, 4]           1,024
           Conv2d-96            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-97            [-1, 512, 4, 4]           1,024
       BasicBlock-98            [-1, 512, 4, 4]               0
           Conv2d-99            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-100            [-1, 512, 4, 4]           1,024
          Conv2d-101            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-102            [-1, 512, 4, 4]           1,024
      BasicBlock-103            [-1, 512, 4, 4]               0
          Conv2d-104            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-105            [-1, 512, 4, 4]           1,024
          Conv2d-106            [-1, 512, 4, 4]       2,359,296
     BatchNorm2d-107            [-1, 512, 4, 4]           1,024
      BasicBlock-108            [-1, 512, 4, 4]               0
AdaptiveAvgPool2d-109            [-1, 512, 1, 1]               0
         Dropout-110                  [-1, 512]               0
          Linear-111                  [-1, 100]          51,300
================================================================
Total params: 30,032,292
Trainable params: 30,032,292
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 25.32
Params size (MB): 114.56
Estimated Total Size (MB): 139.90
----------------------------------------------------------------
Mixed precision training enabled (AMP) for faster training

## Training Hyperparameters
- Epochs: 200
- Batch Size: 128
- Optimizer: SGD (momentum=0.9, weight_decay=5e-4, nesterov=True)
- Scheduler: OneCycleLR (max_lr=0.1, initial_lr=0.004, final_lr=0.0001)
- Label Smoothing: 0.1
- Gradient Clipping: max_norm=1.0
- Mixed Precision: Enabled
- CutMix: Enabled (prob=0.3, alpha=1.0)
- Augmentation: PadIfNeeded(40x40)→RandomCrop(32x32), HFlip, ShiftScaleRotate, CoarseDropout/GaussNoise


## Training Progress

| Epoch | Train Acc | Test Acc | Test Loss | LR | Status |
|-------|-----------|----------|-----------|-----|--------|

### EPOCH: 0

Test set: Average loss: 4.4051, Accuracy: 372/10000 (3.72%)

Current LR: 0.004148

======================================================================
EPOCH:   0 | Train Accuracy:   1.75% | Test Accuracy:   3.72%
======================================================================

|     0 |     1.75% |    3.72% |   4.4051 | 0.004148 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_0.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 1

Test set: Average loss: 4.0679, Accuracy: 794/10000 (7.94%)

Current LR: 0.004591

======================================================================
EPOCH:   1 | Train Accuracy:   3.67% | Test Accuracy:   7.94%
======================================================================

|     1 |     3.67% |    7.94% |   4.0679 | 0.004591 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_1.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 2

Test set: Average loss: 3.7950, Accuracy: 1140/10000 (11.40%)

Current LR: 0.005326

======================================================================
EPOCH:   2 | Train Accuracy:   6.86% | Test Accuracy:  11.40%
======================================================================

|     2 |     6.86% |   11.40% |   3.7950 | 0.005326 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_2.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 3

Test set: Average loss: 3.6198, Accuracy: 1429/10000 (14.29%)

Current LR: 0.006350

======================================================================
EPOCH:   3 | Train Accuracy:   9.73% | Test Accuracy:  14.29%
======================================================================

|     3 |     9.73% |   14.29% |   3.6198 | 0.006350 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_3.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 4

Test set: Average loss: 3.4531, Accuracy: 1739/10000 (17.39%)

Current LR: 0.007654

======================================================================
EPOCH:   4 | Train Accuracy:  12.05% | Test Accuracy:  17.39%
======================================================================

|     4 |    12.05% |   17.39% |   3.4531 | 0.007654 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_4.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_4.pt

### EPOCH: 5

Test set: Average loss: 3.3172, Accuracy: 2081/10000 (20.81%)

Current LR: 0.009232

======================================================================
EPOCH:   5 | Train Accuracy:  14.29% | Test Accuracy:  20.81%
======================================================================

|     5 |    14.29% |   20.81% |   3.3172 | 0.009232 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_5.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 6

Test set: Average loss: 3.1251, Accuracy: 2375/10000 (23.75%)

Current LR: 0.011074

======================================================================
EPOCH:   6 | Train Accuracy:  17.62% | Test Accuracy:  23.75%
======================================================================

|     6 |    17.62% |   23.75% |   3.1251 | 0.011074 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_6.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 7

Test set: Average loss: 2.9668, Accuracy: 2625/10000 (26.25%)

Current LR: 0.013168

======================================================================
EPOCH:   7 | Train Accuracy:  19.22% | Test Accuracy:  26.25%
======================================================================

|     7 |    19.22% |   26.25% |   2.9668 | 0.013168 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_7.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 8

Test set: Average loss: 2.8584, Accuracy: 2841/10000 (28.41%)

Current LR: 0.015502

======================================================================
EPOCH:   8 | Train Accuracy:  22.28% | Test Accuracy:  28.41%
======================================================================

|     8 |    22.28% |   28.41% |   2.8584 | 0.015502 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_8.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 9

Test set: Average loss: 2.7472, Accuracy: 3085/10000 (30.85%)

Current LR: 0.018061

======================================================================
EPOCH:   9 | Train Accuracy:  24.02% | Test Accuracy:  30.85%
======================================================================

|     9 |    24.02% |   30.85% |   2.7472 | 0.018061 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_9.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_9.pt

### EPOCH: 10

Test set: Average loss: 2.5281, Accuracy: 3483/10000 (34.83%)

Current LR: 0.020829

======================================================================
EPOCH:  10 | Train Accuracy:  27.76% | Test Accuracy:  34.83%
======================================================================

|    10 |    27.76% |   34.83% |   2.5281 | 0.020829 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_10.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 11

Test set: Average loss: 2.5140, Accuracy: 3539/10000 (35.39%)

Current LR: 0.023789

======================================================================
EPOCH:  11 | Train Accuracy:  29.81% | Test Accuracy:  35.39%
======================================================================

|    11 |    29.81% |   35.39% |   2.5140 | 0.023789 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_11.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 12

Test set: Average loss: 2.2512, Accuracy: 4088/10000 (40.88%)

Current LR: 0.026923

======================================================================
EPOCH:  12 | Train Accuracy:  33.02% | Test Accuracy:  40.88%
======================================================================

|    12 |    33.02% |   40.88% |   2.2512 | 0.026923 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_12.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 13

Test set: Average loss: 2.2980, Accuracy: 3976/10000 (39.76%)

Current LR: 0.030211

======================================================================
EPOCH:  13 | Train Accuracy:  34.24% | Test Accuracy:  39.76%
======================================================================

|    13 |    34.24% |   39.76% |   2.2980 | 0.030211 |  |

### EPOCH: 14

Test set: Average loss: 2.2158, Accuracy: 4143/10000 (41.43%)

Current LR: 0.033635

======================================================================
EPOCH:  14 | Train Accuracy:  36.73% | Test Accuracy:  41.43%
======================================================================

|    14 |    36.73% |   41.43% |   2.2158 | 0.033635 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_14.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_14.pt

### EPOCH: 15

Test set: Average loss: 2.0701, Accuracy: 4538/10000 (45.38%)

Current LR: 0.037171

======================================================================
EPOCH:  15 | Train Accuracy:  39.17% | Test Accuracy:  45.38%
======================================================================

|    15 |    39.17% |   45.38% |   2.0701 | 0.037171 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_15.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 16

Test set: Average loss: 2.0280, Accuracy: 4651/10000 (46.51%)

Current LR: 0.040799

======================================================================
EPOCH:  16 | Train Accuracy:  40.38% | Test Accuracy:  46.51%
======================================================================

|    16 |    40.38% |   46.51% |   2.0280 | 0.040799 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_16.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 17

Test set: Average loss: 2.1052, Accuracy: 4511/10000 (45.11%)

Current LR: 0.044495

======================================================================
EPOCH:  17 | Train Accuracy:  42.25% | Test Accuracy:  45.11%
======================================================================

|    17 |    42.25% |   45.11% |   2.1052 | 0.044495 |  |

### EPOCH: 18

Test set: Average loss: 2.1847, Accuracy: 4432/10000 (44.32%)

Current LR: 0.048239

======================================================================
EPOCH:  18 | Train Accuracy:  43.19% | Test Accuracy:  44.32%
======================================================================

|    18 |    43.19% |   44.32% |   2.1847 | 0.048239 |  |

### EPOCH: 19

Test set: Average loss: 2.1859, Accuracy: 4371/10000 (43.71%)

Current LR: 0.052005

======================================================================
EPOCH:  19 | Train Accuracy:  44.69% | Test Accuracy:  43.71%
======================================================================

|    19 |    44.69% |   43.71% |   2.1859 | 0.052005 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_19.pt

### EPOCH: 20

Test set: Average loss: 2.1698, Accuracy: 4364/10000 (43.64%)

Current LR: 0.055771

======================================================================
EPOCH:  20 | Train Accuracy:  45.07% | Test Accuracy:  43.64%
======================================================================

|    20 |    45.07% |   43.64% |   2.1698 | 0.055771 |  |

### EPOCH: 21

Test set: Average loss: 2.1774, Accuracy: 4492/10000 (44.92%)

Current LR: 0.059514

======================================================================
EPOCH:  21 | Train Accuracy:  47.93% | Test Accuracy:  44.92%
======================================================================

|    21 |    47.93% |   44.92% |   2.1774 | 0.059514 |  |

### EPOCH: 22

Test set: Average loss: 1.8432, Accuracy: 5128/10000 (51.28%)

Current LR: 0.063211

======================================================================
EPOCH:  22 | Train Accuracy:  46.20% | Test Accuracy:  51.28%
======================================================================

|    22 |    46.20% |   51.28% |   1.8432 | 0.063211 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_22.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 23

Test set: Average loss: 1.7890, Accuracy: 5374/10000 (53.74%)

Current LR: 0.066838

======================================================================
EPOCH:  23 | Train Accuracy:  47.80% | Test Accuracy:  53.74%
======================================================================

|    23 |    47.80% |   53.74% |   1.7890 | 0.066838 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_23.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 24

Test set: Average loss: 1.8992, Accuracy: 5060/10000 (50.60%)

Current LR: 0.070374

======================================================================
EPOCH:  24 | Train Accuracy:  49.87% | Test Accuracy:  50.60%
======================================================================

|    24 |    49.87% |   50.60% |   1.8992 | 0.070374 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_24.pt

### EPOCH: 25

Test set: Average loss: 1.6866, Accuracy: 5568/10000 (55.68%)

Current LR: 0.073797

======================================================================
EPOCH:  25 | Train Accuracy:  49.75% | Test Accuracy:  55.68%
======================================================================

|    25 |    49.75% |   55.68% |   1.6866 | 0.073797 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_25.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 26

Test set: Average loss: 1.7272, Accuracy: 5497/10000 (54.97%)

Current LR: 0.077085

======================================================================
EPOCH:  26 | Train Accuracy:  50.06% | Test Accuracy:  54.97%
======================================================================

|    26 |    50.06% |   54.97% |   1.7272 | 0.077085 |  |

### EPOCH: 27

Test set: Average loss: 1.7303, Accuracy: 5448/10000 (54.48%)

Current LR: 0.080219

======================================================================
EPOCH:  27 | Train Accuracy:  48.17% | Test Accuracy:  54.48%
======================================================================

|    27 |    48.17% |   54.48% |   1.7303 | 0.080219 |  |

### EPOCH: 28

Test set: Average loss: 1.8233, Accuracy: 5439/10000 (54.39%)

Current LR: 0.083179

======================================================================
EPOCH:  28 | Train Accuracy:  49.34% | Test Accuracy:  54.39%
======================================================================

|    28 |    49.34% |   54.39% |   1.8233 | 0.083179 |  |

### EPOCH: 29

Test set: Average loss: 1.7883, Accuracy: 5292/10000 (52.92%)

Current LR: 0.085946

======================================================================
EPOCH:  29 | Train Accuracy:  49.98% | Test Accuracy:  52.92%
======================================================================

|    29 |    49.98% |   52.92% |   1.7883 | 0.085946 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_29.pt

### EPOCH: 30

Test set: Average loss: 1.7603, Accuracy: 5449/10000 (54.49%)

Current LR: 0.088504

======================================================================
EPOCH:  30 | Train Accuracy:  52.48% | Test Accuracy:  54.49%
======================================================================

|    30 |    52.48% |   54.49% |   1.7603 | 0.088504 |  |

### EPOCH: 31

Test set: Average loss: 1.6261, Accuracy: 5727/10000 (57.27%)

Current LR: 0.090837

======================================================================
EPOCH:  31 | Train Accuracy:  52.12% | Test Accuracy:  57.27%
======================================================================

|    31 |    52.12% |   57.27% |   1.6261 | 0.090837 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_31.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 32

Test set: Average loss: 1.6199, Accuracy: 5760/10000 (57.60%)

Current LR: 0.092931

======================================================================
EPOCH:  32 | Train Accuracy:  52.45% | Test Accuracy:  57.60%
======================================================================

|    32 |    52.45% |   57.60% |   1.6199 | 0.092931 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_32.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 33

Test set: Average loss: 1.8148, Accuracy: 5332/10000 (53.32%)

Current LR: 0.094772

======================================================================
EPOCH:  33 | Train Accuracy:  53.36% | Test Accuracy:  53.32%
======================================================================

|    33 |    53.36% |   53.32% |   1.8148 | 0.094772 |  |

### EPOCH: 34

Test set: Average loss: 1.8705, Accuracy: 5299/10000 (52.99%)

Current LR: 0.096349

======================================================================
EPOCH:  34 | Train Accuracy:  53.99% | Test Accuracy:  52.99%
======================================================================

|    34 |    53.99% |   52.99% |   1.8705 | 0.096349 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_34.pt

### EPOCH: 35

Test set: Average loss: 1.7790, Accuracy: 5414/10000 (54.14%)

Current LR: 0.097653

======================================================================
EPOCH:  35 | Train Accuracy:  54.67% | Test Accuracy:  54.14%
======================================================================

|    35 |    54.67% |   54.14% |   1.7790 | 0.097653 |  |

### EPOCH: 36

Test set: Average loss: 1.7109, Accuracy: 5585/10000 (55.85%)

Current LR: 0.098676

======================================================================
EPOCH:  36 | Train Accuracy:  53.48% | Test Accuracy:  55.85%
======================================================================

|    36 |    53.48% |   55.85% |   1.7109 | 0.098676 |  |

### EPOCH: 37

Test set: Average loss: 1.8069, Accuracy: 5268/10000 (52.68%)

Current LR: 0.099410

======================================================================
EPOCH:  37 | Train Accuracy:  54.01% | Test Accuracy:  52.68%
======================================================================

|    37 |    54.01% |   52.68% |   1.8069 | 0.099410 |  |

### EPOCH: 38

Test set: Average loss: 1.6290, Accuracy: 5800/10000 (58.00%)

Current LR: 0.099853

======================================================================
EPOCH:  38 | Train Accuracy:  53.75% | Test Accuracy:  58.00%
======================================================================

|    38 |    53.75% |   58.00% |   1.6290 | 0.099853 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_38.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 39

Test set: Average loss: 1.6461, Accuracy: 5742/10000 (57.42%)

Current LR: 0.100000

======================================================================
EPOCH:  39 | Train Accuracy:  54.97% | Test Accuracy:  57.42%
======================================================================

|    39 |    54.97% |   57.42% |   1.6461 | 0.100000 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_39.pt

### EPOCH: 40

Test set: Average loss: 1.6736, Accuracy: 5724/10000 (57.24%)

Current LR: 0.099990

======================================================================
EPOCH:  40 | Train Accuracy:  56.39% | Test Accuracy:  57.24%
======================================================================

|    40 |    56.39% |   57.24% |   1.6736 | 0.099990 |  |

### EPOCH: 41

Test set: Average loss: 1.7428, Accuracy: 5578/10000 (55.78%)

Current LR: 0.099961

======================================================================
EPOCH:  41 | Train Accuracy:  53.65% | Test Accuracy:  55.78%
======================================================================

|    41 |    53.65% |   55.78% |   1.7428 | 0.099961 |  |

### EPOCH: 42

Test set: Average loss: 1.6597, Accuracy: 5756/10000 (57.56%)

Current LR: 0.099913

======================================================================
EPOCH:  42 | Train Accuracy:  55.32% | Test Accuracy:  57.56%
======================================================================

|    42 |    55.32% |   57.56% |   1.6597 | 0.099913 |  |

### EPOCH: 43

Test set: Average loss: 1.6315, Accuracy: 5783/10000 (57.83%)

Current LR: 0.099846

======================================================================
EPOCH:  43 | Train Accuracy:  54.05% | Test Accuracy:  57.83%
======================================================================

|    43 |    54.05% |   57.83% |   1.6315 | 0.099846 |  |

### EPOCH: 44

Test set: Average loss: 1.8991, Accuracy: 5212/10000 (52.12%)

Current LR: 0.099759

======================================================================
EPOCH:  44 | Train Accuracy:  55.57% | Test Accuracy:  52.12%
======================================================================

|    44 |    55.57% |   52.12% |   1.8991 | 0.099759 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_44.pt

### EPOCH: 45

Test set: Average loss: 1.6413, Accuracy: 5835/10000 (58.35%)

Current LR: 0.099653

======================================================================
EPOCH:  45 | Train Accuracy:  57.33% | Test Accuracy:  58.35%
======================================================================

|    45 |    57.33% |   58.35% |   1.6413 | 0.099653 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_45.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 46

Test set: Average loss: 1.6710, Accuracy: 5723/10000 (57.23%)

Current LR: 0.099528

======================================================================
EPOCH:  46 | Train Accuracy:  56.67% | Test Accuracy:  57.23%
======================================================================

|    46 |    56.67% |   57.23% |   1.6710 | 0.099528 |  |

### EPOCH: 47

Test set: Average loss: 1.6488, Accuracy: 5784/10000 (57.84%)

Current LR: 0.099384

======================================================================
EPOCH:  47 | Train Accuracy:  57.37% | Test Accuracy:  57.84%
======================================================================

|    47 |    57.37% |   57.84% |   1.6488 | 0.099384 |  |

### EPOCH: 48

Test set: Average loss: 1.5903, Accuracy: 5885/10000 (58.85%)

Current LR: 0.099221

======================================================================
EPOCH:  48 | Train Accuracy:  56.29% | Test Accuracy:  58.85%
======================================================================

|    48 |    56.29% |   58.85% |   1.5903 | 0.099221 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_48.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 49

Test set: Average loss: 1.7128, Accuracy: 5685/10000 (56.85%)

Current LR: 0.099039

======================================================================
EPOCH:  49 | Train Accuracy:  55.10% | Test Accuracy:  56.85%
======================================================================

|    49 |    55.10% |   56.85% |   1.7128 | 0.099039 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_49.pt

### EPOCH: 50

Test set: Average loss: 1.6204, Accuracy: 5894/10000 (58.94%)

Current LR: 0.098838

======================================================================
EPOCH:  50 | Train Accuracy:  57.31% | Test Accuracy:  58.94%
======================================================================

|    50 |    57.31% |   58.94% |   1.6204 | 0.098838 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_50.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 51

Test set: Average loss: 1.6831, Accuracy: 5708/10000 (57.08%)

Current LR: 0.098618

======================================================================
EPOCH:  51 | Train Accuracy:  56.47% | Test Accuracy:  57.08%
======================================================================

|    51 |    56.47% |   57.08% |   1.6831 | 0.098618 |  |

### EPOCH: 52

Test set: Average loss: 1.6166, Accuracy: 5916/10000 (59.16%)

Current LR: 0.098379

======================================================================
EPOCH:  52 | Train Accuracy:  57.21% | Test Accuracy:  59.16%
======================================================================

|    52 |    57.21% |   59.16% |   1.6166 | 0.098379 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_52.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 53

Test set: Average loss: 1.7385, Accuracy: 5635/10000 (56.35%)

Current LR: 0.098122

======================================================================
EPOCH:  53 | Train Accuracy:  56.80% | Test Accuracy:  56.35%
======================================================================

|    53 |    56.80% |   56.35% |   1.7385 | 0.098122 |  |

### EPOCH: 54

Test set: Average loss: 1.8380, Accuracy: 5414/10000 (54.14%)

Current LR: 0.097846

======================================================================
EPOCH:  54 | Train Accuracy:  56.67% | Test Accuracy:  54.14%
======================================================================

|    54 |    56.67% |   54.14% |   1.8380 | 0.097846 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_54.pt

### EPOCH: 55

Test set: Average loss: 1.4913, Accuracy: 6144/10000 (61.44%)

Current LR: 0.097552

======================================================================
EPOCH:  55 | Train Accuracy:  57.76% | Test Accuracy:  61.44%
======================================================================

|    55 |    57.76% |   61.44% |   1.4913 | 0.097552 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_55.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 56

Test set: Average loss: 1.5359, Accuracy: 6151/10000 (61.51%)

Current LR: 0.097240

======================================================================
EPOCH:  56 | Train Accuracy:  57.20% | Test Accuracy:  61.51%
======================================================================

|    56 |    57.20% |   61.51% |   1.5359 | 0.097240 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_56.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 57

Test set: Average loss: 1.5457, Accuracy: 6106/10000 (61.06%)

Current LR: 0.096909

======================================================================
EPOCH:  57 | Train Accuracy:  57.11% | Test Accuracy:  61.06%
======================================================================

|    57 |    57.11% |   61.06% |   1.5457 | 0.096909 |  |

### EPOCH: 58

Test set: Average loss: 1.5982, Accuracy: 5913/10000 (59.13%)

Current LR: 0.096560

======================================================================
EPOCH:  58 | Train Accuracy:  58.18% | Test Accuracy:  59.13%
======================================================================

|    58 |    58.18% |   59.13% |   1.5982 | 0.096560 |  |

### EPOCH: 59

Test set: Average loss: 1.6197, Accuracy: 5880/10000 (58.80%)

Current LR: 0.096193

======================================================================
EPOCH:  59 | Train Accuracy:  58.33% | Test Accuracy:  58.80%
======================================================================

|    59 |    58.33% |   58.80% |   1.6197 | 0.096193 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_59.pt

### EPOCH: 60

Test set: Average loss: 1.5539, Accuracy: 6027/10000 (60.27%)

Current LR: 0.095809

======================================================================
EPOCH:  60 | Train Accuracy:  58.14% | Test Accuracy:  60.27%
======================================================================

|    60 |    58.14% |   60.27% |   1.5539 | 0.095809 |  |

### EPOCH: 61

Test set: Average loss: 1.6198, Accuracy: 5862/10000 (58.62%)

Current LR: 0.095406

======================================================================
EPOCH:  61 | Train Accuracy:  56.51% | Test Accuracy:  58.62%
======================================================================

|    61 |    56.51% |   58.62% |   1.6198 | 0.095406 |  |

### EPOCH: 62

Test set: Average loss: 1.5965, Accuracy: 5958/10000 (59.58%)

Current LR: 0.094987

======================================================================
EPOCH:  62 | Train Accuracy:  59.07% | Test Accuracy:  59.58%
======================================================================

|    62 |    59.07% |   59.58% |   1.5965 | 0.094987 |  |

### EPOCH: 63

Test set: Average loss: 1.5691, Accuracy: 6008/10000 (60.08%)

Current LR: 0.094549

======================================================================
EPOCH:  63 | Train Accuracy:  57.23% | Test Accuracy:  60.08%
======================================================================

|    63 |    57.23% |   60.08% |   1.5691 | 0.094549 |  |

### EPOCH: 64

Test set: Average loss: 1.5292, Accuracy: 6119/10000 (61.19%)

Current LR: 0.094095

======================================================================
EPOCH:  64 | Train Accuracy:  59.22% | Test Accuracy:  61.19%
======================================================================

|    64 |    59.22% |   61.19% |   1.5292 | 0.094095 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_64.pt

### EPOCH: 65

Test set: Average loss: 1.5934, Accuracy: 5975/10000 (59.75%)

Current LR: 0.093624

======================================================================
EPOCH:  65 | Train Accuracy:  57.13% | Test Accuracy:  59.75%
======================================================================

|    65 |    57.13% |   59.75% |   1.5934 | 0.093624 |  |

### EPOCH: 66

Test set: Average loss: 1.5687, Accuracy: 6015/10000 (60.15%)

Current LR: 0.093136

======================================================================
EPOCH:  66 | Train Accuracy:  59.20% | Test Accuracy:  60.15%
======================================================================

|    66 |    59.20% |   60.15% |   1.5687 | 0.093136 |  |

### EPOCH: 67

Test set: Average loss: 1.5475, Accuracy: 6007/10000 (60.07%)

Current LR: 0.092631

======================================================================
EPOCH:  67 | Train Accuracy:  61.10% | Test Accuracy:  60.07%
======================================================================

|    67 |    61.10% |   60.07% |   1.5475 | 0.092631 |  |

### EPOCH: 68

Test set: Average loss: 1.5829, Accuracy: 5947/10000 (59.47%)

Current LR: 0.092110

======================================================================
EPOCH:  68 | Train Accuracy:  58.98% | Test Accuracy:  59.47%
======================================================================

|    68 |    58.98% |   59.47% |   1.5829 | 0.092110 |  |

### EPOCH: 69

Test set: Average loss: 1.5285, Accuracy: 6064/10000 (60.64%)

Current LR: 0.091572

======================================================================
EPOCH:  69 | Train Accuracy:  59.90% | Test Accuracy:  60.64%
======================================================================

|    69 |    59.90% |   60.64% |   1.5285 | 0.091572 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_69.pt

### EPOCH: 70

Test set: Average loss: 1.5925, Accuracy: 6004/10000 (60.04%)

Current LR: 0.091019

======================================================================
EPOCH:  70 | Train Accuracy:  59.41% | Test Accuracy:  60.04%
======================================================================

|    70 |    59.41% |   60.04% |   1.5925 | 0.091019 |  |

### EPOCH: 71

Test set: Average loss: 1.5350, Accuracy: 6072/10000 (60.72%)

Current LR: 0.090450

======================================================================
EPOCH:  71 | Train Accuracy:  59.93% | Test Accuracy:  60.72%
======================================================================

|    71 |    59.93% |   60.72% |   1.5350 | 0.090450 |  |

### EPOCH: 72

Test set: Average loss: 1.5101, Accuracy: 6173/10000 (61.73%)

Current LR: 0.089865

======================================================================
EPOCH:  72 | Train Accuracy:  59.14% | Test Accuracy:  61.73%
======================================================================

|    72 |    59.14% |   61.73% |   1.5101 | 0.089865 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_72.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 73

Test set: Average loss: 1.4801, Accuracy: 6189/10000 (61.89%)

Current LR: 0.089265

======================================================================
EPOCH:  73 | Train Accuracy:  59.81% | Test Accuracy:  61.89%
======================================================================

|    73 |    59.81% |   61.89% |   1.4801 | 0.089265 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_73.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 74

Test set: Average loss: 1.7068, Accuracy: 5690/10000 (56.90%)

Current LR: 0.088649

======================================================================
EPOCH:  74 | Train Accuracy:  60.26% | Test Accuracy:  56.90%
======================================================================

|    74 |    60.26% |   56.90% |   1.7068 | 0.088649 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_74.pt

### EPOCH: 75

Test set: Average loss: 1.4692, Accuracy: 6233/10000 (62.33%)

Current LR: 0.088019

======================================================================
EPOCH:  75 | Train Accuracy:  60.53% | Test Accuracy:  62.33%
======================================================================

|    75 |    60.53% |   62.33% |   1.4692 | 0.088019 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_75.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 76

Test set: Average loss: 1.5589, Accuracy: 6140/10000 (61.40%)

Current LR: 0.087374

======================================================================
EPOCH:  76 | Train Accuracy:  61.29% | Test Accuracy:  61.40%
======================================================================

|    76 |    61.29% |   61.40% |   1.5589 | 0.087374 |  |

### EPOCH: 77

Test set: Average loss: 1.4738, Accuracy: 6305/10000 (63.05%)

Current LR: 0.086715

======================================================================
EPOCH:  77 | Train Accuracy:  59.32% | Test Accuracy:  63.05%
======================================================================

|    77 |    59.32% |   63.05% |   1.4738 | 0.086715 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_77.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 78

Test set: Average loss: 1.4663, Accuracy: 6256/10000 (62.56%)

Current LR: 0.086041

======================================================================
EPOCH:  78 | Train Accuracy:  60.52% | Test Accuracy:  62.56%
======================================================================

|    78 |    60.52% |   62.56% |   1.4663 | 0.086041 |  |

### EPOCH: 79

Test set: Average loss: 1.5533, Accuracy: 6056/10000 (60.56%)

Current LR: 0.085354

======================================================================
EPOCH:  79 | Train Accuracy:  58.84% | Test Accuracy:  60.56%
======================================================================

|    79 |    58.84% |   60.56% |   1.5533 | 0.085354 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_79.pt

### EPOCH: 80

Test set: Average loss: 1.3943, Accuracy: 6458/10000 (64.58%)

Current LR: 0.084653

======================================================================
EPOCH:  80 | Train Accuracy:  61.12% | Test Accuracy:  64.58%
======================================================================

|    80 |    61.12% |   64.58% |   1.3943 | 0.084653 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_80.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 81

Test set: Average loss: 1.4477, Accuracy: 6404/10000 (64.04%)

Current LR: 0.083939

======================================================================
EPOCH:  81 | Train Accuracy:  59.66% | Test Accuracy:  64.04%
======================================================================

|    81 |    59.66% |   64.04% |   1.4477 | 0.083939 |  |

### EPOCH: 82

Test set: Average loss: 1.5500, Accuracy: 6077/10000 (60.77%)

Current LR: 0.083211

======================================================================
EPOCH:  82 | Train Accuracy:  60.02% | Test Accuracy:  60.77%
======================================================================

|    82 |    60.02% |   60.77% |   1.5500 | 0.083211 |  |

### EPOCH: 83

Test set: Average loss: 1.5841, Accuracy: 6029/10000 (60.29%)

Current LR: 0.082471

======================================================================
EPOCH:  83 | Train Accuracy:  61.04% | Test Accuracy:  60.29%
======================================================================

|    83 |    61.04% |   60.29% |   1.5841 | 0.082471 |  |

### EPOCH: 84

Test set: Average loss: 1.5792, Accuracy: 6060/10000 (60.60%)

Current LR: 0.081718

======================================================================
EPOCH:  84 | Train Accuracy:  61.23% | Test Accuracy:  60.60%
======================================================================

|    84 |    61.23% |   60.60% |   1.5792 | 0.081718 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_84.pt

### EPOCH: 85

Test set: Average loss: 1.6799, Accuracy: 5774/10000 (57.74%)

Current LR: 0.080953

======================================================================
EPOCH:  85 | Train Accuracy:  61.86% | Test Accuracy:  57.74%
======================================================================

|    85 |    61.86% |   57.74% |   1.6799 | 0.080953 |  |

### EPOCH: 86

Test set: Average loss: 1.6298, Accuracy: 5906/10000 (59.06%)

Current LR: 0.080177

======================================================================
EPOCH:  86 | Train Accuracy:  60.65% | Test Accuracy:  59.06%
======================================================================

|    86 |    60.65% |   59.06% |   1.6298 | 0.080177 |  |

### EPOCH: 87

Test set: Average loss: 1.5282, Accuracy: 6128/10000 (61.28%)

Current LR: 0.079388

======================================================================
EPOCH:  87 | Train Accuracy:  61.24% | Test Accuracy:  61.28%
======================================================================

|    87 |    61.24% |   61.28% |   1.5282 | 0.079388 |  |

### EPOCH: 88

Test set: Average loss: 1.4577, Accuracy: 6303/10000 (63.03%)

Current LR: 0.078588

======================================================================
EPOCH:  88 | Train Accuracy:  60.38% | Test Accuracy:  63.03%
======================================================================

|    88 |    60.38% |   63.03% |   1.4577 | 0.078588 |  |

### EPOCH: 89

Test set: Average loss: 1.4213, Accuracy: 6379/10000 (63.79%)

Current LR: 0.077777

======================================================================
EPOCH:  89 | Train Accuracy:  60.33% | Test Accuracy:  63.79%
======================================================================

|    89 |    60.33% |   63.79% |   1.4213 | 0.077777 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_89.pt

### EPOCH: 90

Test set: Average loss: 1.5795, Accuracy: 6007/10000 (60.07%)

Current LR: 0.076956

======================================================================
EPOCH:  90 | Train Accuracy:  60.43% | Test Accuracy:  60.07%
======================================================================

|    90 |    60.43% |   60.07% |   1.5795 | 0.076956 |  |

### EPOCH: 91

Test set: Average loss: 1.4791, Accuracy: 6264/10000 (62.64%)

Current LR: 0.076124

======================================================================
EPOCH:  91 | Train Accuracy:  62.28% | Test Accuracy:  62.64%
======================================================================

|    91 |    62.28% |   62.64% |   1.4791 | 0.076124 |  |

### EPOCH: 92

Test set: Average loss: 1.5622, Accuracy: 6094/10000 (60.94%)

Current LR: 0.075282

======================================================================
EPOCH:  92 | Train Accuracy:  61.54% | Test Accuracy:  60.94%
======================================================================

|    92 |    61.54% |   60.94% |   1.5622 | 0.075282 |  |

### EPOCH: 93

Test set: Average loss: 1.4864, Accuracy: 6340/10000 (63.40%)

Current LR: 0.074430

======================================================================
EPOCH:  93 | Train Accuracy:  60.83% | Test Accuracy:  63.40%
======================================================================

|    93 |    60.83% |   63.40% |   1.4864 | 0.074430 |  |

### EPOCH: 94

Test set: Average loss: 1.4155, Accuracy: 6345/10000 (63.45%)

Current LR: 0.073569

======================================================================
EPOCH:  94 | Train Accuracy:  61.81% | Test Accuracy:  63.45%
======================================================================

|    94 |    61.81% |   63.45% |   1.4155 | 0.073569 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_94.pt

### EPOCH: 95

Test set: Average loss: 1.3792, Accuracy: 6452/10000 (64.52%)

Current LR: 0.072698

======================================================================
EPOCH:  95 | Train Accuracy:  59.87% | Test Accuracy:  64.52%
======================================================================

|    95 |    59.87% |   64.52% |   1.3792 | 0.072698 |  |

### EPOCH: 96

Test set: Average loss: 1.4850, Accuracy: 6231/10000 (62.31%)

Current LR: 0.071819

======================================================================
EPOCH:  96 | Train Accuracy:  60.85% | Test Accuracy:  62.31%
======================================================================

|    96 |    60.85% |   62.31% |   1.4850 | 0.071819 |  |

### EPOCH: 97

Test set: Average loss: 1.5258, Accuracy: 6114/10000 (61.14%)

Current LR: 0.070932

======================================================================
EPOCH:  97 | Train Accuracy:  61.45% | Test Accuracy:  61.14%
======================================================================

|    97 |    61.45% |   61.14% |   1.5258 | 0.070932 |  |

### EPOCH: 98

Test set: Average loss: 1.4452, Accuracy: 6310/10000 (63.10%)

Current LR: 0.070036

======================================================================
EPOCH:  98 | Train Accuracy:  60.40% | Test Accuracy:  63.10%
======================================================================

|    98 |    60.40% |   63.10% |   1.4452 | 0.070036 |  |

### EPOCH: 99

Test set: Average loss: 1.5336, Accuracy: 6157/10000 (61.57%)

Current LR: 0.069133

======================================================================
EPOCH:  99 | Train Accuracy:  61.80% | Test Accuracy:  61.57%
======================================================================

|    99 |    61.80% |   61.57% |   1.5336 | 0.069133 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_99.pt

### EPOCH: 100

Test set: Average loss: 1.5148, Accuracy: 6215/10000 (62.15%)

Current LR: 0.068222

======================================================================
EPOCH: 100 | Train Accuracy:  60.45% | Test Accuracy:  62.15%
======================================================================

|   100 |    60.45% |   62.15% |   1.5148 | 0.068222 |  |

### EPOCH: 101

Test set: Average loss: 1.4811, Accuracy: 6225/10000 (62.25%)

Current LR: 0.067305

======================================================================
EPOCH: 101 | Train Accuracy:  64.06% | Test Accuracy:  62.25%
======================================================================

|   101 |    64.06% |   62.25% |   1.4811 | 0.067305 |  |

### EPOCH: 102

Test set: Average loss: 1.4847, Accuracy: 6284/10000 (62.84%)

Current LR: 0.066380

======================================================================
EPOCH: 102 | Train Accuracy:  62.52% | Test Accuracy:  62.84%
======================================================================

|   102 |    62.52% |   62.84% |   1.4847 | 0.066380 |  |

### EPOCH: 103

Test set: Average loss: 1.4674, Accuracy: 6312/10000 (63.12%)

Current LR: 0.065450

======================================================================
EPOCH: 103 | Train Accuracy:  61.91% | Test Accuracy:  63.12%
======================================================================

|   103 |    61.91% |   63.12% |   1.4674 | 0.065450 |  |

### EPOCH: 104

Test set: Average loss: 1.4060, Accuracy: 6442/10000 (64.42%)

Current LR: 0.064513

======================================================================
EPOCH: 104 | Train Accuracy:  62.72% | Test Accuracy:  64.42%
======================================================================

|   104 |    62.72% |   64.42% |   1.4060 | 0.064513 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_104.pt

### EPOCH: 105

Test set: Average loss: 1.4726, Accuracy: 6328/10000 (63.28%)

Current LR: 0.063571

======================================================================
EPOCH: 105 | Train Accuracy:  61.44% | Test Accuracy:  63.28%
======================================================================

|   105 |    61.44% |   63.28% |   1.4726 | 0.063571 |  |

### EPOCH: 106

Test set: Average loss: 1.4966, Accuracy: 6224/10000 (62.24%)

Current LR: 0.062624

======================================================================
EPOCH: 106 | Train Accuracy:  61.32% | Test Accuracy:  62.24%
======================================================================

|   106 |    61.32% |   62.24% |   1.4966 | 0.062624 |  |

### EPOCH: 107

Test set: Average loss: 1.4335, Accuracy: 6361/10000 (63.61%)

Current LR: 0.061671

======================================================================
EPOCH: 107 | Train Accuracy:  63.18% | Test Accuracy:  63.61%
======================================================================

|   107 |    63.18% |   63.61% |   1.4335 | 0.061671 |  |

### EPOCH: 108

Test set: Average loss: 1.5018, Accuracy: 6272/10000 (62.72%)

Current LR: 0.060715

======================================================================
EPOCH: 108 | Train Accuracy:  62.00% | Test Accuracy:  62.72%
======================================================================

|   108 |    62.00% |   62.72% |   1.5018 | 0.060715 |  |

### EPOCH: 109

Test set: Average loss: 1.4225, Accuracy: 6423/10000 (64.23%)

Current LR: 0.059754

======================================================================
EPOCH: 109 | Train Accuracy:  60.17% | Test Accuracy:  64.23%
======================================================================

|   109 |    60.17% |   64.23% |   1.4225 | 0.059754 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_109.pt

### EPOCH: 110

Test set: Average loss: 1.3344, Accuracy: 6583/10000 (65.83%)

Current LR: 0.058789

======================================================================
EPOCH: 110 | Train Accuracy:  62.64% | Test Accuracy:  65.83%
======================================================================

|   110 |    62.64% |   65.83% |   1.3344 | 0.058789 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_110.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 111

Test set: Average loss: 1.4982, Accuracy: 6257/10000 (62.57%)

Current LR: 0.057821

======================================================================
EPOCH: 111 | Train Accuracy:  63.44% | Test Accuracy:  62.57%
======================================================================

|   111 |    63.44% |   62.57% |   1.4982 | 0.057821 |  |

### EPOCH: 112

Test set: Average loss: 1.4222, Accuracy: 6440/10000 (64.40%)

Current LR: 0.056850

======================================================================
EPOCH: 112 | Train Accuracy:  62.29% | Test Accuracy:  64.40%
======================================================================

|   112 |    62.29% |   64.40% |   1.4222 | 0.056850 |  |

### EPOCH: 113

Test set: Average loss: 1.3421, Accuracy: 6616/10000 (66.16%)

Current LR: 0.055876

======================================================================
EPOCH: 113 | Train Accuracy:  62.79% | Test Accuracy:  66.16%
======================================================================

|   113 |    62.79% |   66.16% |   1.3421 | 0.055876 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_113.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 114

Test set: Average loss: 1.3059, Accuracy: 6665/10000 (66.65%)

Current LR: 0.054900

======================================================================
EPOCH: 114 | Train Accuracy:  63.73% | Test Accuracy:  66.65%
======================================================================

|   114 |    63.73% |   66.65% |   1.3059 | 0.054900 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_114.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_114.pt

### EPOCH: 115

Test set: Average loss: 1.3606, Accuracy: 6584/10000 (65.84%)

Current LR: 0.053922

======================================================================
EPOCH: 115 | Train Accuracy:  63.84% | Test Accuracy:  65.84%
======================================================================

|   115 |    63.84% |   65.84% |   1.3606 | 0.053922 |  |

### EPOCH: 116

Test set: Average loss: 1.4264, Accuracy: 6364/10000 (63.64%)

Current LR: 0.052943

======================================================================
EPOCH: 116 | Train Accuracy:  64.38% | Test Accuracy:  63.64%
======================================================================

|   116 |    64.38% |   63.64% |   1.4264 | 0.052943 |  |

### EPOCH: 117

Test set: Average loss: 1.3114, Accuracy: 6752/10000 (67.52%)

Current LR: 0.051962

======================================================================
EPOCH: 117 | Train Accuracy:  62.09% | Test Accuracy:  67.52%
======================================================================

|   117 |    62.09% |   67.52% |   1.3114 | 0.051962 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_117.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 118

Test set: Average loss: 1.3751, Accuracy: 6555/10000 (65.55%)

Current LR: 0.050981

======================================================================
EPOCH: 118 | Train Accuracy:  65.61% | Test Accuracy:  65.55%
======================================================================

|   118 |    65.61% |   65.55% |   1.3751 | 0.050981 |  |

### EPOCH: 119

Test set: Average loss: 1.3277, Accuracy: 6707/10000 (67.07%)

Current LR: 0.049999

======================================================================
EPOCH: 119 | Train Accuracy:  64.37% | Test Accuracy:  67.07%
======================================================================

|   119 |    64.37% |   67.07% |   1.3277 | 0.049999 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_119.pt

### EPOCH: 120

Test set: Average loss: 1.3751, Accuracy: 6510/10000 (65.10%)

Current LR: 0.049018

======================================================================
EPOCH: 120 | Train Accuracy:  63.54% | Test Accuracy:  65.10%
======================================================================

|   120 |    63.54% |   65.10% |   1.3751 | 0.049018 |  |

### EPOCH: 121

Test set: Average loss: 1.4233, Accuracy: 6446/10000 (64.46%)

Current LR: 0.048037

======================================================================
EPOCH: 121 | Train Accuracy:  65.96% | Test Accuracy:  64.46%
======================================================================

|   121 |    65.96% |   64.46% |   1.4233 | 0.048037 |  |

### EPOCH: 122

Test set: Average loss: 1.3115, Accuracy: 6680/10000 (66.80%)

Current LR: 0.047056

======================================================================
EPOCH: 122 | Train Accuracy:  65.00% | Test Accuracy:  66.80%
======================================================================

|   122 |    65.00% |   66.80% |   1.3115 | 0.047056 |  |

### EPOCH: 123

Test set: Average loss: 1.4128, Accuracy: 6434/10000 (64.34%)

Current LR: 0.046077

======================================================================
EPOCH: 123 | Train Accuracy:  65.57% | Test Accuracy:  64.34%
======================================================================

|   123 |    65.57% |   64.34% |   1.4128 | 0.046077 |  |

### EPOCH: 124

Test set: Average loss: 1.3712, Accuracy: 6548/10000 (65.48%)

Current LR: 0.045099

======================================================================
EPOCH: 124 | Train Accuracy:  64.22% | Test Accuracy:  65.48%
======================================================================

|   124 |    64.22% |   65.48% |   1.3712 | 0.045099 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_124.pt

### EPOCH: 125

Test set: Average loss: 1.4065, Accuracy: 6473/10000 (64.73%)

Current LR: 0.044123

======================================================================
EPOCH: 125 | Train Accuracy:  63.78% | Test Accuracy:  64.73%
======================================================================

|   125 |    63.78% |   64.73% |   1.4065 | 0.044123 |  |

### EPOCH: 126

Test set: Average loss: 1.3056, Accuracy: 6730/10000 (67.30%)

Current LR: 0.043149

======================================================================
EPOCH: 126 | Train Accuracy:  65.71% | Test Accuracy:  67.30%
======================================================================

|   126 |    65.71% |   67.30% |   1.3056 | 0.043149 |  |

### EPOCH: 127

Test set: Average loss: 1.2864, Accuracy: 6780/10000 (67.80%)

Current LR: 0.042178

======================================================================
EPOCH: 127 | Train Accuracy:  66.30% | Test Accuracy:  67.80%
======================================================================

|   127 |    66.30% |   67.80% |   1.2864 | 0.042178 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_127.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 128

Test set: Average loss: 1.3393, Accuracy: 6577/10000 (65.77%)

Current LR: 0.041210

======================================================================
EPOCH: 128 | Train Accuracy:  64.74% | Test Accuracy:  65.77%
======================================================================

|   128 |    64.74% |   65.77% |   1.3393 | 0.041210 |  |

### EPOCH: 129

Test set: Average loss: 1.2685, Accuracy: 6880/10000 (68.80%)

Current LR: 0.040245

======================================================================
EPOCH: 129 | Train Accuracy:  66.87% | Test Accuracy:  68.80%
======================================================================

|   129 |    66.87% |   68.80% |   1.2685 | 0.040245 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_129.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_129.pt

### EPOCH: 130

Test set: Average loss: 1.3407, Accuracy: 6670/10000 (66.70%)

Current LR: 0.039285

======================================================================
EPOCH: 130 | Train Accuracy:  64.43% | Test Accuracy:  66.70%
======================================================================

|   130 |    64.43% |   66.70% |   1.3407 | 0.039285 |  |

### EPOCH: 131

Test set: Average loss: 1.3339, Accuracy: 6696/10000 (66.96%)

Current LR: 0.038328

======================================================================
EPOCH: 131 | Train Accuracy:  65.08% | Test Accuracy:  66.96%
======================================================================

|   131 |    65.08% |   66.96% |   1.3339 | 0.038328 |  |

### EPOCH: 132

Test set: Average loss: 1.2850, Accuracy: 6732/10000 (67.32%)

Current LR: 0.037375

======================================================================
EPOCH: 132 | Train Accuracy:  66.34% | Test Accuracy:  67.32%
======================================================================

|   132 |    66.34% |   67.32% |   1.2850 | 0.037375 |  |

### EPOCH: 133

Test set: Average loss: 1.4037, Accuracy: 6568/10000 (65.68%)

Current LR: 0.036428

======================================================================
EPOCH: 133 | Train Accuracy:  66.21% | Test Accuracy:  65.68%
======================================================================

|   133 |    66.21% |   65.68% |   1.4037 | 0.036428 |  |

### EPOCH: 134

Test set: Average loss: 1.2942, Accuracy: 6777/10000 (67.77%)

Current LR: 0.035486

======================================================================
EPOCH: 134 | Train Accuracy:  67.69% | Test Accuracy:  67.77%
======================================================================

|   134 |    67.69% |   67.77% |   1.2942 | 0.035486 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_134.pt

### EPOCH: 135

Test set: Average loss: 1.3244, Accuracy: 6735/10000 (67.35%)

Current LR: 0.034549

======================================================================
EPOCH: 135 | Train Accuracy:  68.28% | Test Accuracy:  67.35%
======================================================================

|   135 |    68.28% |   67.35% |   1.3244 | 0.034549 |  |

### EPOCH: 136

Test set: Average loss: 1.3870, Accuracy: 6587/10000 (65.87%)

Current LR: 0.033619

======================================================================
EPOCH: 136 | Train Accuracy:  66.80% | Test Accuracy:  65.87%
======================================================================

|   136 |    66.80% |   65.87% |   1.3870 | 0.033619 |  |

### EPOCH: 137

Test set: Average loss: 1.2820, Accuracy: 6792/10000 (67.92%)

Current LR: 0.032694

======================================================================
EPOCH: 137 | Train Accuracy:  67.83% | Test Accuracy:  67.92%
======================================================================

|   137 |    67.83% |   67.92% |   1.2820 | 0.032694 |  |

### EPOCH: 138

Test set: Average loss: 1.2476, Accuracy: 6907/10000 (69.07%)

Current LR: 0.031777

======================================================================
EPOCH: 138 | Train Accuracy:  67.03% | Test Accuracy:  69.07%
======================================================================

|   138 |    67.03% |   69.07% |   1.2476 | 0.031777 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_138.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 139

Test set: Average loss: 1.2590, Accuracy: 6883/10000 (68.83%)

Current LR: 0.030866

======================================================================
EPOCH: 139 | Train Accuracy:  67.55% | Test Accuracy:  68.83%
======================================================================

|   139 |    67.55% |   68.83% |   1.2590 | 0.030866 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_139.pt

### EPOCH: 140

Test set: Average loss: 1.2029, Accuracy: 7024/10000 (70.24%)

Current LR: 0.029963

======================================================================
EPOCH: 140 | Train Accuracy:  69.18% | Test Accuracy:  70.24%
======================================================================

|   140 |    69.18% |   70.24% |   1.2029 | 0.029963 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_140.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 141

Test set: Average loss: 1.2512, Accuracy: 6890/10000 (68.90%)

Current LR: 0.029068

======================================================================
EPOCH: 141 | Train Accuracy:  68.51% | Test Accuracy:  68.90%
======================================================================

|   141 |    68.51% |   68.90% |   1.2512 | 0.029068 |  |

### EPOCH: 142

Test set: Average loss: 1.2021, Accuracy: 7010/10000 (70.10%)

Current LR: 0.028180

======================================================================
EPOCH: 142 | Train Accuracy:  70.26% | Test Accuracy:  70.10%
======================================================================

|   142 |    70.26% |   70.10% |   1.2021 | 0.028180 |  |

### EPOCH: 143

Test set: Average loss: 1.2640, Accuracy: 6870/10000 (68.70%)

Current LR: 0.027301

======================================================================
EPOCH: 143 | Train Accuracy:  71.08% | Test Accuracy:  68.70%
======================================================================

|   143 |    71.08% |   68.70% |   1.2640 | 0.027301 |  |

### EPOCH: 144

Test set: Average loss: 1.2487, Accuracy: 6910/10000 (69.10%)

Current LR: 0.026431

======================================================================
EPOCH: 144 | Train Accuracy:  69.72% | Test Accuracy:  69.10%
======================================================================

|   144 |    69.72% |   69.10% |   1.2487 | 0.026431 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_144.pt

### EPOCH: 145

Test set: Average loss: 1.1952, Accuracy: 7044/10000 (70.44%)

Current LR: 0.025570

======================================================================
EPOCH: 145 | Train Accuracy:  69.74% | Test Accuracy:  70.44%
======================================================================

|   145 |    69.74% |   70.44% |   1.1952 | 0.025570 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_145.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 146

Test set: Average loss: 1.1960, Accuracy: 7057/10000 (70.57%)

Current LR: 0.024718

======================================================================
EPOCH: 146 | Train Accuracy:  71.75% | Test Accuracy:  70.57%
======================================================================

|   146 |    71.75% |   70.57% |   1.1960 | 0.024718 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_146.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 147

Test set: Average loss: 1.1992, Accuracy: 7012/10000 (70.12%)

Current LR: 0.023876

======================================================================
EPOCH: 147 | Train Accuracy:  70.53% | Test Accuracy:  70.12%
======================================================================

|   147 |    70.53% |   70.12% |   1.1992 | 0.023876 |  |

### EPOCH: 148

Test set: Average loss: 1.2220, Accuracy: 6990/10000 (69.90%)

Current LR: 0.023044

======================================================================
EPOCH: 148 | Train Accuracy:  69.99% | Test Accuracy:  69.90%
======================================================================

|   148 |    69.99% |   69.90% |   1.2220 | 0.023044 |  |

### EPOCH: 149

Test set: Average loss: 1.2034, Accuracy: 7075/10000 (70.75%)

Current LR: 0.022223

======================================================================
EPOCH: 149 | Train Accuracy:  71.00% | Test Accuracy:  70.75%
======================================================================

|   149 |    71.00% |   70.75% |   1.2034 | 0.022223 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_149.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_149.pt

### EPOCH: 150

Test set: Average loss: 1.1901, Accuracy: 7070/10000 (70.70%)

Current LR: 0.021412

======================================================================
EPOCH: 150 | Train Accuracy:  70.60% | Test Accuracy:  70.70%
======================================================================

|   150 |    70.60% |   70.70% |   1.1901 | 0.021412 |  |

### EPOCH: 151

Test set: Average loss: 1.1618, Accuracy: 7133/10000 (71.33%)

Current LR: 0.020612

======================================================================
EPOCH: 151 | Train Accuracy:  72.00% | Test Accuracy:  71.33%
======================================================================

|   151 |    72.00% |   71.33% |   1.1618 | 0.020612 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_151.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 152

Test set: Average loss: 1.1558, Accuracy: 7166/10000 (71.66%)

Current LR: 0.019823

======================================================================
EPOCH: 152 | Train Accuracy:  71.11% | Test Accuracy:  71.66%
======================================================================

|   152 |    71.11% |   71.66% |   1.1558 | 0.019823 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_152.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 153

Test set: Average loss: 1.2006, Accuracy: 7084/10000 (70.84%)

Current LR: 0.019047

======================================================================
EPOCH: 153 | Train Accuracy:  70.56% | Test Accuracy:  70.84%
======================================================================

|   153 |    70.56% |   70.84% |   1.2006 | 0.019047 |  |

### EPOCH: 154

Test set: Average loss: 1.1625, Accuracy: 7143/10000 (71.43%)

Current LR: 0.018282

======================================================================
EPOCH: 154 | Train Accuracy:  72.68% | Test Accuracy:  71.43%
======================================================================

|   154 |    72.68% |   71.43% |   1.1625 | 0.018282 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_154.pt

### EPOCH: 155

Test set: Average loss: 1.1312, Accuracy: 7232/10000 (72.32%)

Current LR: 0.017529

======================================================================
EPOCH: 155 | Train Accuracy:  73.72% | Test Accuracy:  72.32%
======================================================================

|   155 |    73.72% |   72.32% |   1.1312 | 0.017529 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_155.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 156

Test set: Average loss: 1.1666, Accuracy: 7158/10000 (71.58%)

Current LR: 0.016789

======================================================================
EPOCH: 156 | Train Accuracy:  73.20% | Test Accuracy:  71.58%
======================================================================

|   156 |    73.20% |   71.58% |   1.1666 | 0.016789 |  |

### EPOCH: 157

Test set: Average loss: 1.0925, Accuracy: 7365/10000 (73.65%)

Current LR: 0.016061

======================================================================
EPOCH: 157 | Train Accuracy:  74.70% | Test Accuracy:  73.65%
======================================================================

|   157 |    74.70% |   73.65% |   1.0925 | 0.016061 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_157.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 158

Test set: Average loss: 1.1115, Accuracy: 7315/10000 (73.15%)

Current LR: 0.015347

======================================================================
EPOCH: 158 | Train Accuracy:  75.17% | Test Accuracy:  73.15%
======================================================================

|   158 |    75.17% |   73.15% |   1.1115 | 0.015347 |  |

### EPOCH: 159

Test set: Average loss: 1.1258, Accuracy: 7293/10000 (72.93%)

Current LR: 0.014646

======================================================================
EPOCH: 159 | Train Accuracy:  74.32% | Test Accuracy:  72.93%
======================================================================

|   159 |    74.32% |   72.93% |   1.1258 | 0.014646 |  |
Checkpoint saved: checkpoints/checkpoint_epoch_159.pt

### EPOCH: 160

Test set: Average loss: 1.1119, Accuracy: 7314/10000 (73.14%)

Current LR: 0.013959

======================================================================
EPOCH: 160 | Train Accuracy:  74.66% | Test Accuracy:  73.14%
======================================================================

|   160 |    74.66% |   73.14% |   1.1119 | 0.013959 |  |

### EPOCH: 161

Test set: Average loss: 1.0849, Accuracy: 7298/10000 (72.98%)

Current LR: 0.013286

======================================================================
EPOCH: 161 | Train Accuracy:  74.65% | Test Accuracy:  72.98%
======================================================================

|   161 |    74.65% |   72.98% |   1.0849 | 0.013286 |  |

### EPOCH: 162

Test set: Average loss: 1.0893, Accuracy: 7398/10000 (73.98%)

Current LR: 0.012626

======================================================================
EPOCH: 162 | Train Accuracy:  76.06% | Test Accuracy:  73.98%
======================================================================

|   162 |    76.06% |   73.98% |   1.0893 | 0.012626 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_162.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 163

Test set: Average loss: 1.0450, Accuracy: 7472/10000 (74.72%)

Current LR: 0.011982

======================================================================
EPOCH: 163 | Train Accuracy:  78.76% | Test Accuracy:  74.72%
======================================================================

|   163 |    78.76% |   74.72% |   1.0450 | 0.011982 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_163.pt
Best model saved: checkpoints/best_model.pt

### EPOCH: 164

Test set: Average loss: 1.0495, Accuracy: 7476/10000 (74.76%)

Current LR: 0.011351

======================================================================
EPOCH: 164 | Train Accuracy:  76.03% | Test Accuracy:  74.76%
======================================================================

|   164 |    76.03% |   74.76% |   1.0495 | 0.011351 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_164.pt
Best model saved: checkpoints/best_model.pt
Checkpoint saved: checkpoints/checkpoint_epoch_164.pt

### EPOCH: 165

Test set: Average loss: 1.0518, Accuracy: 7514/10000 (75.14%)

Current LR: 0.010736

======================================================================
EPOCH: 165 | Train Accuracy:  76.24% | Test Accuracy:  75.14%
======================================================================

|   165 |    76.24% |   75.14% |   1.0518 | 0.010736 | 🎯 **BEST** |
Checkpoint saved: checkpoints/checkpoint_epoch_165.pt
Best model saved: checkpoints/best_model.pt

======================================================================
🎯 TARGET ACHIEVED! Test Accuracy: 75.14% >= 75%
======================================================================

Checkpoint saved: checkpoints/checkpoint_epoch_165.pt
Best model saved: checkpoints/best_model.pt

================================================================================
# Training Completed!
**Best Test Accuracy: 75.14%**
Training Session Ended: 2025-10-21 05:39:01
================================================================================


## Generating GradCAM Visualizations
Loading best model for visualization...

✅ Loaded best model (Test Acc: 75.14%)
✅ Using target layer for GradCAM visualization

Generating GradCAM visualizations for 6 sample images...
⚠️  GradCAM visualization failed: GradCAM.__init__() got an unexpected keyword argument 'target_layer'
Continuing with training completion...

================================================================================
