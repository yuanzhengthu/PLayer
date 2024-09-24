## Official implementation of PLayer: A Plug-and-Play Embedded Neural System to Boost Neural Organoid 3D Reconstruction

### Authors: Yuanzheng Ma, Davit Khutsishvili, Zihan Zang, Wei Yue, Zhen Guo, Tao Feng, Zitian Wang, Shaohua Ma*, and Xun Guan*
Neural organoids and confocal microscopy have the potential to play an important role in micro-connectome research and understanding neural patterns. In this study, we present *PLayer*, a Plug-and-Play Embedded Neural System, demonstrating the utilization of sparse confocal microscopy layers to interpolate continuous axial resolution. With an embedded system focused on neural network pruning, image scaling, and post-processing methods, *PLayer* achieves performance metrics with an average Structural Similarity Index of 0.9217 and a Peak Signal-to-Noise Ratio of 27.75 dB, all within 20 seconds. This represents a significant time-saving of 85.71% from simplified image processing. By harnessing statistical map estimation in interpolation and incorporating the Vision Transformer-based *Restorer*, *PLayer* ensures 2D layer consistency while mitigating heavy computational dependence. \textit{PLayer} reconstructs 3D neural organoid confocal data continuously with limited computational power, aiming to translate into wide acceptance of fundamental connectomics and pattern-related research with embedded devices.

![Demo of 3D Display of Organoid Voulmn on Pi5](README/video1.gif)

### How to Use

#### Setting Up the Environment
To prepare your environment, install the required Python packages:
```bash
pip install -r requirements.txt
```
### Model Pruning
#### Execute the following script to prune and fine-tune the model:
```
python model_prune_fine_tune.py
```
### Deploying to Raspberry Pi
#### Download and run the test script on your Raspberry Pi:
```
python test.py
```
