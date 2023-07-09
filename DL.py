import os
import nibabel as nib
from nilearn.datasets import fetch_adhd
from nilearn.decomposition import DictLearning
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import subprocess
import easygui
import csv

N_SUBJECTS = 1 
COMPONENTS = 2

# Define directories for saving files
original_dir = './original_nii_files'
processed_dir = './processed_nii_files'

os.makedirs(original_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

# Fetch the fMRI dataset
rest_dataset = fetch_adhd(n_subjects=N_SUBJECTS)
func_filenames = rest_dataset.func

# Initialize DictLearning object
dict_learn = DictLearning(
    n_components=COMPONENTS,
    smoothing_fwhm=6.0,
    memory="nilearn_cache",
    memory_level=2,
    random_state=0,
    standardize="zscore_sample",
    n_jobs=-1,
)

total_subjects = len(func_filenames)
start_time = time.time()

# Function to fit the data and return the components_img
def process_subject(subject_idx, func_file):
    dict_learn.fit(func_file)

    # Save the original file
    original_file_name = os.path.join(original_dir, f'original_{subject_idx}.nii.gz')
    nib.save(nib.load(func_file), original_file_name)

    components_img = dict_learn.components_img_

    # Open the CSV file for appending
    with open('labels.csv', 'a') as f:
        writer = csv.writer(f)

        # Iterate over each component and save it to a separate file
        for component_idx in range(components_img.shape[-1]):
            component_img = nib.Nifti1Image(components_img.dataobj[..., component_idx], components_img.affine)
            component_file_name = os.path.join(processed_dir, f'processed_{subject_idx}_component_{component_idx + 1}.nii.gz')
            nib.save(component_img, component_file_name)

            # # Open the original and the component image in FSLeyes
            # subprocess.run(['fsleyes', original_file_name, component_file_name])

            # # Create a textbox GUI for entering the component label
            # label = easygui.enterbox("Enter the label for the component:")

            # Open the original and the component image in FSLeyes
            subprocess.Popen(['fsleyes', original_file_name, component_file_name])

            # Create a textbox GUI for entering the component label
            label = easygui.enterbox("Enter the label for the component:")

            # Write the label to the CSV file
            writer.writerow([subject_idx, component_idx, label])

    return components_img

# Parallel processing of subjects
results = Parallel(n_jobs=-1)(
    delayed(process_subject)(subject_idx + 1, func_file)
    for subject_idx, func_file in enumerate(func_filenames)
)

# Create a progress bar
progress_bar = tqdm(total=total_subjects, desc="Processing Subjects")

# Process the results
for subject_idx, components_img in enumerate(results, start=1):
    # Update the progress bar
    progress_bar.update(1)
    progress_bar.set_postfix_str(f"Processed Subject {subject_idx}")

    # Calculate time remaining
    elapsed_time = time.time() - start_time
    time_per_subject = elapsed_time / subject_idx
    subjects_remaining = total_subjects - subject_idx
    time_remaining = subjects_remaining * time_per_subject
    progress_bar.set_postfix({"Time Remaining": f"{time_remaining:.2f} seconds"})

# Close the progress bar
progress_bar.close()
