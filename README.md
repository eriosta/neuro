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

**Option 1:** Using the provided environment.yml file:
```bash
conda env create -f environment.yml
```
**Option 2:** Manually creating the environment:
```bash
conda create -n my_environment_name python=3.10
conda activate my_environment_name
conda install -c conda-forge nilearn=0.10.1 nibabel=5.0.1 joblib=1.1.1 tqdm=4.64.1 pysimplegui=4.60.5
```
### Option 2: Pip Installation
#### Install Python
* [Download Python](https://www.python.org/downloads/)
#### Install the required libraries using pip
```bash
pip install requirements.txt
```

Choose the option that best suits your needs and system configuration. Once you have completed the setup, you will be ready to run the script.

# Usage

Run the script using the following command:

```bash
python app.py <N_SUBJECTS> <N_COMPONENTS>
```
Replace `<N_SUBJECTS>` with the desired number of subjects and `<N_COMPONENTS>` with the desired number of components. For example, to process 5 subjects with 10 components each, use:
```bash
python app.py 5 10
```
The script will fetch the fMRI dataset (ADHD dataset) using the nilearn.datasets module. It will then perform DictLearning on the specified number of subjects and save the resulting components as separate NIfTI files.

During the processing, FSLeyes will be launched to display the original and component images for each subject. You can use the PySimpleGUI dropdown GUI to select the label for each component.

The processed component files will be saved in the processed_nii_files directory, and the labels will be saved in a timestamped CSV file.

Please note that the script assumes the necessary directory structure (original_nii_files and processed_nii_files) exists in the current working directory. If the directories don't exist, the script will create them.

# Disclaimer
This script is provided as-is and without any warranty. Use it at your own risk. Ensure you have sufficient disk space and computational resources to perform the processing for the specified number of subjects and components.