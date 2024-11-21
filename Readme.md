# Towards 3D Virtual Try-On: Clothing Reconstruction and Retargeting with Gaussian Splatting

### Authors:
- [Andrea Sanchietti](andreus00.github.io)
- [Co-Advisor: Riccardo Marin](ricma.netlify.app)
- [Advisor: Emanuele Rodolà](https://gladia.di.uniroma1.it/authors/rodola/)


### Introduction
This is the work that I developed as my thesis for the Master's Degree in Computer Science at the University of Rome La Sapienza. The goal of this work is to develop a pipeline that allows to reconstruct the 3D shape of a person from multiview images, and then perform a virtual try-on of a garment on the reconstructed person. You can find my thesis in the `thesis` folder.

### Installation
To install the dependencies, you can create an environment from the `environment.yml` file. You can do this by running the following command:
```bash
conda env create -f environment.yml
```

### Data
You need the **4D-Dress** dataset, obtainable at this [link](https://eth-ait.github.io/4d-dress/).

Once you have downloaded the dataset (or even part of it), you need to move it under the `4D-Dress` folder.

Finally, you have to run the `evaluation/prepare_reconstruction_evaluation_data.py` script to prepare the data for the evaluation.

### SMPL
You need the SMPL model, which can be downloaded from the [SMPL website](https://smpl.is.tue.mpg.de/). Once you have downloaded the model, you need to move it under the `smpl` folder. Notice that you will need `smpl/smpl`, `smpl/smpl300`, and `smpl/smplx`.

### NICP and Gaussian Splatting
You can download the repositories from the respective github repos:
- My fork of [NICP](https://github.com/Andreus00/NICP)
- My fork of [Gaussian Splatting](https://github.com/Andreus00/gaussian-splatting)

Follow the instructions in the respective repositories to install the dependencies and the missing files.

### Perform the reconstruction and retargeting
To perform the reconstruction between two arbitrary avatars, you can run the `train_baseline_on_user.py` script on each of them, and then use the `run_retarget_lbs_on_user.py` script to retarget the garment from one avatar to the other. Examples of the usage of these scripts can be found in the `start.sh` script.

### Evaluation
To run the evaluation, use the `run_evaluation.py` script.
