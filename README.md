# **MaxwellNet**
This repository is the official code implementation of the paper, "MaxwellNet: Physics-driven deep neural network training based on Maxwell’s equations" by [Joowon Lim](https://www.linkedin.com/in/joowon-lim/) and [Demetri Psaltis](https://scholar.google.com/citations?&user=-CVR2h8AAAAJ). You can refer to the following materials for the details of implementation,
- [Main article](https://aip.scitation.org/doi/10.1063/5.0071616 "Main article")
- [Supplementary material](https://aip.scitation.org/doi/suppl/10.1063/5.0071616 "Supplementary material")

Also, we had an interview on this work, 
- [Scilight interview](https://aip.scitation.org/doi/full/10.1063/10.0009285 "Scilight interview")

### **Overall scheme and idea**
The novelty of this work is to train a deep neural network, MaxwellNet, which solves Maxwell's equations using physics-driven loss. In other words, we are using the residual of Maxwell's equations as a loss function to train MaxwellNet, therefore, it does not require ground truth solutions to train it. Furthermore, we utilized MaxwellNet in a novel inverse design scheme, and we encourage you to refer to the [main article](https://aip.scitation.org/doi/10.1063/5.0071616 "Main article") for details.
<br />

![Scheme](/figures/scheme.png)
 
 <br />



## **Installation**
Our code is based on Windows 10, pytorch 1.7.1, CUDA 11.0, and python 3.7.
We recommend using conda for installation.

```
conda env create --file environment.yaml
conda activate maxwellnet
```

## **Run**

### **1. MaxwellNet Training**
```
python train_maxwellnet.py --directory <YOUR_DIRECTORY>
```
In <YOUR_DIRECTORY>, you need to have 'train.npz' which contains the training dataset and 'specs_maxwell.json' where you specify training parameters. A brief description of the parameters can be found below. I encourage you to read the [supplementary material](https://aip.scitation.org/doi/suppl/10.1063/5.0071616 "supplementary material") to understand the parameters.

| NetworkSpecs | Description |
| :---: | :--- |
| depth [int] | Depth of UNet. |
| filter [int] | Channel numbers in the first layer of UNet. |
| norm [str] | Type of normalization ('weight' for weight normalization, 'batch' for batch normalization, and 'no' for no normalization). |
| up_mode [str] | Upsample mode of UNet (either 'upcov' for transpose convolution or 'upsample' for upsampling). |


| PhysicalSpecs | Description |
| :---: | :--- |
| wavelength [float] | Wavelength in [um]. |
| dpl [int] | One pixel size is 'wavelength / dpl' [um]. |
| Nx [int] | Pixel number along the x-axis. This is equivalent to the pixel number along the x-axis of your scattering sample.|
| Nz [int] | Pixel number along the z-axis (light propagation direction). This is equivalent to the pixel number along the z-axis of your scattering sample. |
| pml_thickness [int] | Perfectly-matched-layer (PML) thickness in pixel number. 'pml_thickness * wavelength / dpl' is the actual thickness of PML layer in micrometers. |
| symmetry_x [bool] | If this is True, MaxwellNet will assume your input scattering sample is symmetric along the x-axis. For example, when given a sample whose Nx and Nz are 100 and 200, respectively, if this sample is symmetric along the x-axis, you can save only half of it (Nx=50, Nz=200) in your train file (train.npz) and set 'symmetry_x' as True. |
| mode [str] | 'te' or 'tm' (Transverse Electric or Transverse Magnetic). |
| high_order [str] | 'second' or 'fourth'. It decides which order (second or fourth order) to calculate the gradient. 'fourth' is more accurate than 'second'. |


#### **Examples**
*Training for a single spheric lens.*

If you just want to train a model for a single lens (which would be a good exercise as it runs for a short time), you can train MaxwellNet for a single spheric lens as followings,
* TE mode.
  ```
  python train_maxwellnet.py --directory examples\spheric_te
  ```
* TM mode.
  ```
  python train_maxwellnet.py --directory examples\spheric_tm
  ```

*Training for multiple lenses.*

You can download the datasets of multiple lenses [here](https://drive.google.com/drive/folders/1ZXPKntdBQUOyMYvmKM7Ol6woCN2Rsrqj?usp=sharing). Download and place 'lens_te' and 'lens_tm' folders under 'examples' folder.
* Transverse Electric (TE) mode.
  ```
  python train_maxwellnet.py --directory examples\lens_te
  ```
* Transverse Magnetic (TM) mode.
  ```
  python train_maxwellnet.py --directory examples\lens_tm
  ```
The above training cases take about 37 (TE mode) and 63 (TM mode) hours on V100, respectively.

<br />

### **2. MaxwellNet Solution**
If you want to check the solution found by MaxwellNet, 

```
python solution_maxwellnet.py --directory <YOUR_DIRECTORY> --model_filename <YOUR_MODEL_FILENAME> --sample_filename <YOUR_SAMPLE_FILENAME>
```
It will provide the sample (<YOUR_SAMPLE_FILENAME> in <YOUR_DIRECTORY>) to the saved model (<YOUR_MODEL_FILENAME>) and return the solution found by MaxwellNet, and this output will be saved as an image in <YOUR_DIRECTORY> as you can see in the below examples.

#### **Examples**
If you want to calculate the solution found by MaxwellNet for the single spheric lenses (as trained above),
* TE mode.
  ```
  python solution_maxwellnet.py --directory examples\spheric_te --model_filename 250000_te_fourth.pt --sample_filename sample.npz
  ```
* TM mode.
  ```
  python solution_maxwellnet.py --directory examples\spheric_tm --model_filename 250000_tm_fourth.pt --sample_filename sample.npz
  ```

  | Mode | Result |
  | :---: | :---: |
  |  TE mode  |![Scheme](/figures/te_result.png) |
  |  TM mode  |![Scheme](/figures/tm_result.png) |
You can find the solutions for the multiple lens training cases similarly.

## **Citation**

If you find our work useful in your research, please consider citing our paper:
```
@article{lim2022maxwellnet,
  title={MaxwellNet: Physics-driven deep neural network training based on Maxwell’s equations},
  author={Lim, Joowon and Psaltis, Demetri},
  journal={APL Photonics},
  volume={7},
  number={1},
  pages={011301},
  year={2022},
  publisher={AIP Publishing LLC}
}
```

## **Acknowledgments**
We referred to the code from the following repo, [UNet](https://github.com/jvanvugt/pytorch-unet). We thank the authors for sharing their code.