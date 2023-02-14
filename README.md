# bayesian-learning-for-receivers

*”A Bayesian version will usually make things better.”* 

-- [Andrew Gelman, Columbia University](http://www.stat.columbia.edu/~gelman/book/gelman_quotes.pdf). 

# Bayesian Learning for Deep Receivers

Python repository for the paper "Modular Model-Based Bayesian Learning for Uncertainty-Aware and Reliable Deep MIMO Receivers".

Please cite our [paper](https://arxiv.org/pdf/2302.02436.pdf), if the code is used for publishing research.

# Table of Contents

- [Introduction](#introduction)
- [Folders Structure](#folders-structure)
  * [python_code](#python_code)
    + [channel](#channel)
    + [detectors](#detectors)
    + [plotters](#plotters)
    + [utils](#utils)
  * [resources](#resources)
  * [dir_definitions](#dir_definitions)
- [Execution](#execution)
  * [Environment Installation](#environment-installation)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This repository implements the proposed model-based Bayesian framework for [DeepSIC](https://arxiv.org/abs/2002.03214), a machine-learning model-based MIMO detector. We explain on the different directories and subdirectories below.

# Folders Structure

## python_code 

The python simulations of the simplified communication chain: symbols generation, channel transmission and detection.

### channel 

Includes the symbols generation and transmission part, up to the creation of the dataset composed of (transmitted, received) tuples in the channel_dataset wrapper class. The modulation is done in the modulator file.

### detectors 

Includes the next files:

(1) The backbone trainer.py which holds the most basic functions, including the network initialization and the sequential transmission in the channel and BER calculation. 

(2) The DeepSIC trainer and backbone detector, including the Bayesian variants. Note that we included end-to-end implementation of DeepSIC even as it is not employed in the paper (DeepSIC in the paper refers to the sequential trained one which has higher performance).

(3) The black-box DNN we compare to, and its Bayesian variant for comparison - again it is not used in the paper as the DNN is inferior to DeepSIC in small data regime (so its Bayesian variant is not so interesting).

### plotters

The main script is plotter_main.py, and it is used to plot the figures in the paper including ser versus snr, and reliability diagrams.

### utils

Extra utils for many different things: 

* python utils - saving and loading pkls. 

* metrics - calculating accuracy, confidence, ECE and sampling frequency for reliability diagrams.

* config_singleton - holds the singleton definition of the config yaml.

* probs utils - for generate symbols from states; symbols from probs and vice versa.

* bayesian utils - for the calculation of the LBD loss.

### config.yaml

Controls all hyperparameters:

* seed - random integer used as the generation seed.

* channel_type - only 'MIMO' is supported for the conference version.

* channel_model - the type of channel used, only 'Synthetic' is support in the conference version but we will add in the journal paper more.

* detector_type - the type of evaluted detector. 'seq_model' - sequentially trained DeepSIC, 'end_to_end_model' - end-to-end trained DeepSIC, 
'model_based_bayesian'- our proposed model-based Bayesian DeepSIC, 'bayesian' - Bayesian DeepSIC, 'black_box' - DNN detector, 'bayesian_black_box' - Bayesian black-box DNN detector.

* linear - only linear channel is supported in this version.

* fading_in_channel - whether the channel is time-varying. We used 'False' such that the channel is static for the conference paper.

* snr - signal-to-noise value in dB (float).

* modulation_type - which modulation to use, in the set of ['BPSK','QPSK','EightPSK'].
 
* n_user - integer number of transmitting devices.

* n_ant - integer number of received signals.

* block_length - number of total bits in transmission (pilots + info).

* pilot_size - number of pilot bits in the transmission.

* blocks_num - number of blocks to transmit.

* is_online_training - whether to train at each incoming block using its pilot part or skip training. 

* loss_type - loss type in the set ['BCE','CrossEntropy','MSE'].

* optimizer_type - in the set ['Adam','RMSprop','SGD'].

### evaluate

Run the evaluation using one of the methods, as appears in config.yaml

## resources

Keeps the configs runs files for creating the paper's figures.

## dir_definitions 

Definitions of relative directories.

# Execution

To execute the code, first download and install Git, Anaconda and PyCharm.

Then install the environment, follow the installation setup below. 

At last, open PyCharm in the root directory. You may run either the trainers or one of the plotters.

This code was simulated with GeForce RTX 3060 with driver version 516.94 and CUDA 11.6. 

## Environment Installation

1. Open git bash and cd to a working directory of you choice.

2. Clone this repository to your local machine.

3. Open Anaconda prompt and navigate to the cloned repository.

4. Run the command "conda env create -f environment.yml". This should install the required python environment.

5. Open the cloned directory using PyCharm.

6. After the project has been opened in PyCharm, go to settings, File -> Settings... (or CTRL ALT S)

7. In the opened window open the tab Project -> Project Interpreter

8. In the new window, click on the cog icon and then on Add...

9. In the add python interpreter window, click on the Conda Environment tab

10. Select Existing environment and navigate to where the python.exe executable of the deep_ensemble environment is installed under the interpreter setting

  - For windows its usually found at C:\users\<username>\anaconda3\envs\environment\python.exe)

  - For linux its usually found at /home/<username>/anaconda3
  
11. Click OK

12. Done!
