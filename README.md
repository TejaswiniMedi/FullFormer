# FullFormer
This is official repository of FullFormer: Generating Shapes Inside Shapes paper. This work is accepted to DAGM/GCPR2023.
## The Methodology of this paper is as follows:
<img src="Images/method1-1.png" width="900">

## Install:
The project requires a Linux system that is equipped with Cuda 10.

All subsequent commands assume that you have cloned the repository in your terminal and navigated to its location.

A file named "env.yml" contains all necessary python dependencies.

To conveniently install them automatically with [anaconda](https://www.anaconda.com/) you can use:

```
conda env create -f env.yml

conda activate VQDIG
```
## Data
To replicate our experiments, please download the corresponding raw [ShapeNet data](https://shapenet.org/) 
For FullCars dataset mentioned in paper: [Full Cars](https://www.dropbox.com/scl/fi/pc3j5firmi4rxkm3rl1oy/FullFormer.tar.gz?rlkey=pau3v5hosy68p13scmylhsywv&dl=0)

## Experimental Preparation:
For processing raw data for our model 
```
python preprocess.py 
```
To split the random train/validation/test split of data
```
python dataprocessing/create_split.py
```
## Reconstruction
To train autoencoder of our model
```
python train.py
```
To generate reconstruction results
```
python generate.py
```
## Generation
To train the transformer to generate latent codes, which are learned during reconstruction
```
python training_transformer.py
```
To generate generation results
```
python latent generation.py
```
## Contact
For questions and comments please contact [Tejaswini Medi](tejaswini.medi@uni-siegen.de) via mail


