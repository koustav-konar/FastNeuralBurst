# FastNeuralBurst
FastNeuralBurst is python package for cosmological inference with likelihood-free or Simulation-based Inference using the dispersion measure (DM) of Fast Radio Bursts (FRBs). The package is constructed using functions, which makes it flexible and modular for any input models. The config file contains all the parameters with a brief desccription, which acts as the input.

## Installation and Examples
First, clone the directory using:
```shell
git clone git@github.com:koustav-konar/FastNeuralBursts.git
```
or
```shell
git clone https://github.com/koustav-konar/FastNeuralBursts.git
```
Then, navigate to the cloned directory
```shell
cd FastNeuralBurst
conda env create -f conda_env.yaml
conda activate fnb_env
```
You will also need to install two additional packages. First one is ```PyTorch```. Please choose according to your system specification. The second one is the ```LtU ILI``` package, which also offers installation options.
```

[ltu-ili.readthedocs.io](https://ltu-ili.readthedocs.io/en/latest/)
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)
```
Once all the external packages are installed, the program can be run using the configuration file named ``config_SBI_DM_FRB.ini``, where all parameters are stored and explained. Please refer to the jupyter notebook for an example of the whole setup.
