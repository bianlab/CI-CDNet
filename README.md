# Complex-domain enhacing neural network for large-scale coherent imaging

This repository contains the code for complex-domain neural nwtwork (CI-CDNet). For more information, please contact Liheng Bian (bian at bit dot edu dot cn).

## Abstract
Large-scale computational imaging allows an unprecedented resolution beyond the space-bandwidth production (SBP) limit of the optical system. In coherent imaging (CI), the joint reconstruction of amplitude and phase further expands the information throughput and sheds light on label-free observation of biological samples at micro or even nano level. The existing large-scale CI techniques usually require scanning/modulation multiple times to guarantee measurement diversity and long exposure time to achieve a high signal-to-noise ratio (SNR). Such cumbersome procedure restricts clinic applications for rapid and low-phototoxicity cell imaging. In this work, a complex-domain enhancing neural network for large-scale coherent imaging termed CI-CDNet is reported for various large-scale CI modalities with outstanding reconstruction image quality and efficiency. CI-CDNet is able to exploit the latent coupling information between amplitude and phase, realizing multi-dimensional representation of complex wavefront. The cross-field characterization framework empowers strong generalization and robustness for various coherent modalities, allowing high-quality and efficient imaging under extremely few data volume and low exposure time. We applied CI-CDNet in various large-scale CI modalities including Kramers-Kronig-relations holography, Fourier ptychographic microscopy, and lensless coded ptychography. A series of simulations and experiments validate that CI-CDNet can reduce data volume requirement and exposure time by more than one order of magnitude. We further demonstrated that the high-quality reconstruction of CI-CDNet benefits the subsequent high-level semantic analysis.


## Usage

Please clone this repository by Git or download the zip file firstly. 

### main_quickly_test

Run `main_quickly_test.py` file to start a quick demo of CI-CDNet.

Run `main_test_addnoise.py` file to start different noise levels. 

## Download pre-trained model
Baidu Netdisk address:  https://pan.baidu.com/s/1gYxIVyMNK55agnv8lkmFkQ 

Password：aaab

## Requirements and Dependencies
python == 3.7.11

pytorch == 1.8.1
