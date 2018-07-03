# Ising Super Resolution
A Keras implementation of Convolutional Neural Networks (CNNs) for super-resolving classical Ising configurations sampled with standard Monte Carlo methods.

*This repository is still under construction*. I update as I am cleaning up the code used for the actual project.

It is possible to change the default loading/saving directories in data/directories.py file.

## Required modules
Tensorflow, Keras, argparse, os, numpy, scipy.stats (for linear regression)

## Main Scripts

***1) train.py***: Trains a model. Depending on the `-CR` options this can be on critical temperature data or data in different temperatures. The metrics during training are saved as an `.npy` file and the Keras model graph as an `.h5` file.

#### Settings

- Basic information:

`-CR`: `True` if data at critical temperature are used. `False` for data in different temperatures.

`-RGWD`: Use well-defined block spin RG for tied blocks. This assigns the sign of the upper left square. If False the sign is assigned randomly.

`-L`: Linear size of the *output* configuration. The input is L/2.

`-Tind`: Temperature indices to train on. The default `T_list` is defined in `directories.py`. It is not required to train on all temperatures every time we run the training script. Used only if `-CR` is `False`. If `[]` all temperatures are used.

- Sample numbers:

`-nTR`: Number of samples in the training data set. This is the number that appears in the file name.

`-nTE`: Number of samples in the test data set. This is the number that appears in the file name.

`-TRS`: Number of samples that we finally use for training.

`-VALS`: Number of samples that we finally use for validation (from the test data set).

- Architecture Hyperparameters:

`-VER`: Version to appear in model names in order to save multiple models with the same hyperparameters. Used only when `-CR` is `True`.

`-PBC`: Use periodic boundary conditions for padding.

`-ACT`: Activation function to use in hidden layers. Give string: can use any keras activation. Activation in the last layer is by default sigmoid.

`-HF`: List of hidden filters. Last layer filter is by default 1.

`-K`: List of kernels. Must have length one more than `-HF`.

- Compiling Hyperparameters:

`-OPT`: Optimizer. Use any keras optimizer

`-CE`: If `True` it uses cross-entropy as loss. If `False` it uses mean squared error.

`-magR`:  Coefficient of the magnetization regularization in loss.

`-enR`: Coefficient of the energy regularization in loss.

- Training Hyperparameters

`-EP`: Number of epochs for training. If `-ES` is `True` this is ignored and turned to 1000.

`-BS`: Batch size for training. Also included in model name.

`-ES`: If `True` use keras early stopping. If `False` train for `-EP` epochs.

`-ESpat`: Early stopping patience. Only when `-ES` is `True`.

`-ESdelta`: Early stopping delta. Only when `-ES` is `True`.

`-VB`: Keras verbose for messages during training.

***2) test.py:*** Tests a trained network by calculating thermodynamic quantities from the predicted output. It loads a trained keras model and use it to predict and then calculate quantities. The results are saved in `.npy` file with the format: (*temperatures*, *output interpretation*, *quantity*).

- *temperatures*: The different temperatures. If `-CR` is `False` this is ommited.

- *output intepretation*: Five values: 0=Original MC, 1=Decimated RG, 2=Continuous SR, 3=Rounded SR, 4=Sampled SR.

- *quantity*: 0=Magnetization, 1=Energy, 2=Susceptibility, 3=Heat Capacity, 4=Mag^2, 5=Mag^4, 6=En^2, 7=TPF(L/4), 8=TPF(L/2), 9=S0, 10=S1, 11=S2. TPF = Two-point function, S=Fourier transform of TPF (see https://arxiv.org/abs/1101.3281).

#### Settings
Some settings are the same with `train.py` and are not repeated here.

`-Mind`: Model index in the `listdir` listing.

`-OUT`: If `True` it saves the predicted continuous output in `.npy` files.

`-Tind`: Do the calculation for specific temperatures. If `[]` the calculation is done in all temperatures.

***3) list_models.py:*** Lists trained models using `listdir`. This can be used to find the index required when we run `test.py`.

***4) train_multiple_exponents.py:*** Runs the training for critical configurations and calculates critical exponents. The calculation is run multiple times according to `-C` setting. The `.h5` graph is saved for every trained network and an `.npy` file with predicted observables. The format of this file is ... .

#### Settings
Some settings are the same with `train.py` and are not repeated here.

`-C`: Number of times to run training and calculation.

`-UP`: Number of upsamplings for the critical exponent calculation.

`-PRreg`: If `True` print regression results after every calculation.

`-TPF`: If `True` calculate two-point function in observables.

`-CORR`: If `True` calculate correlation length.
