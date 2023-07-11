# Dictionary Learning and GUI-based Labeling
This repository contains code that demonstrates how to perform dictionary learning on functional MRI (fMRI) data using the Nilearn library. The code utilizes Nilearn's DictLearning class to learn a dictionary of components from fMRI data and saves the resulting components as separate NIfTI files.

## Dictionary Learning
Dictionary learning is a technique used to decompose fMRI data into spatial components and their corresponding time courses. The process involves initializing a dictionary, sparse coding to obtain coefficients, updating the dictionary, and iterating until convergence. The learned components provide insights into brain activity patterns.

## Prerequisites
Before running the code, ensure that you have the following dependencies installed:

`python` (version 3.6 or higher)
`numpy`
`nibabel`
`nilearn`
`joblib`
`tqdm`
`subprocess`
`easygui`
`csv`

You can install these dependencies using `pip`:
```bash
pip install numpy nibabel nilearn joblib tqdm
```

## Usage
Follow the instructions below to use the code:

1. Clone or download this repository to your local machine.
2. Navigate to the project directory:
```bash
cd neuro
```
3. Update the desired values for the following variables in the code:
  - `N_SUBJECTS`: The number of subjects to process.
  - `COMPONENTS`: The number of components to learn.
  - `original_dir`: The directory to save the original NIfTI files.
  - `processed_dir`: The directory to save the processed component NIfTI files.
5. Run the script:
```bash
python DL.py
```
The script will perform the following steps:
1. Fetch the fMRI dataset using Nilearn.
2. Initialize the DictLearning object with the specified parameters.
3. Process each subject's fMRI data in parallel:
   - Fit the data to learn the components using dictionary learninng.
   - Save the original NIfTI file.
   - Save each component as a separate NIfTI file.
   - Prompt for a label for each component using a GUI (`easygui`).
   - Write the subject index, component index, and label to a CSV file.
6. After the script completes, you will find the original and processed NIfTI files in the respective directories specified.
