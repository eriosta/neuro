# Dictionary Learning and GUI-based Labeling
This script performs fMRI data processing using the DictLearning algorithm from the nilearn package. It allows you to specify the number of subjects and components as command-line arguments.

## Getting Started
```bash
eri@EGOTOWER % python app.py --help
```
```
usage: test.py [-h] [-s] n_subjects n_components

positional arguments:
  n_subjects    number of subjects
  n_components  number of components

options:
  -h, --help    show this help message and exit
  -s, --save    save '.nii' files
```
```bash
python app.py <N_SUBJECTS> <N_COMPONENTS>
```

```bash
python app.py 30 200
```
# Prerequisites
To set up the environment for running the script, you have two options: creating a conda environment or installing the required libraries using pip.

## Step 1

### Install FMRIB Software Library (FSL)
* [Download FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki)

## Step 2

### Option 1: Conda Environment
#### Install Miniconda or Anaconda
* [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [Download Anaconda](https://www.anaconda.com/products/individual)

#### Create a conda environment

```bash
conda env create -f environment.yml
```

### Option 2: Pip Installation
#### Install Python
* [Download Python](https://www.python.org/downloads/)
#### Install the required libraries using pip
```bash
pip install requirements.txt
```

# Usage

Run the script using the following command:

```bash
python app.py <N_SUBJECTS> <N_COMPONENTS>
```
Replace `<N_SUBJECTS>` with the desired number of subjects and `<N_COMPONENTS>` with the desired number of components. For example, to process 5 subjects with 10 components each, use:
```bash
python app.py 5 10
```
# Disclaimer
This script is provided as-is and without any warranty. Use it at your own risk. Ensure you have sufficient disk space and computational resources to perform the processing for the specified number of subjects and components.