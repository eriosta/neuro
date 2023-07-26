import os
import nibabel as nib
from nilearn.datasets import fetch_adhd
from nilearn.decomposition import DictLearning, CanICA
import time
import subprocess
import argparse
import numpy as np
from scipy.stats import zscore

class NiftiViewer:
    def __init__(self, n_subjects, n_components, method, smoothing_fwhm=None, memory=None, memory_level=None, random_state=None, standardize=None, n_jobs=None):
        self.n_subjects = n_subjects
        self.n_components = n_components
        self.method = method
        self.smoothing_fwhm = smoothing_fwhm if smoothing_fwhm is not None else 6.0
        self.memory = memory if memory is not None else "nilearn_cache"
        self.memory_level = memory_level if memory_level is not None else 2
        self.random_state = random_state if random_state is not None else 0
        self.standardize = standardize if standardize is not None else "zscore_sample"
        self.n_jobs = n_jobs if n_jobs is not None else -1
        self.original_dir = './original_nii_files'
        self.processed_dir = './processed_nii_files'
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.func_filenames = fetch_adhd(n_subjects=self.n_subjects).func
        self.total_subjects = len(self.func_filenames)
        self.decomposition_object = None

    def set_decomposition_object(self):
        if self.method == 'dict_learning':
            self.decomposition_object = DictLearning(n_components=self.n_components,
                                                     smoothing_fwhm=self.smoothing_fwhm,
                                                     memory=self.memory,
                                                     memory_level=self.memory_level,
                                                     random_state=self.random_state,
                                                     standardize=self.standardize,
                                                     n_jobs=self.n_jobs)
        elif self.method == 'ica':
            self.decomposition_object = CanICA(n_components=self.n_components,
                                               smoothing_fwhm=self.smoothing_fwhm,
                                               memory=self.memory,
                                               memory_level=self.memory_level,
                                               random_state=self.random_state,
                                               standardize=self.standardize,
                                               n_jobs=self.n_jobs)
        else:
            raise ValueError("Invalid method. Choose 'dict_learning' or 'ica'.")

    def process_subject(self, subject_idx, func_file):
        self.decomposition_object.fit(func_file)

        # Save the original file
        original_file_name = os.path.join(self.original_dir, f'original_{subject_idx}.nii.gz')
        nib.save(nib.load(func_file), original_file_name)

        components_img = self.decomposition_object.components_img_

        # Iterate over each component and save it to a separate file
        for component_idx in range(components_img.shape[-1]):
            component_img = nib.Nifti1Image(
                components_img.dataobj[..., component_idx], components_img.affine
            )
            component_file_name = os.path.join(
                self.processed_dir, f'processed_{subject_idx}_component_{component_idx + 1}.nii.gz'
            )
            nib.save(component_img, component_file_name)

        return True

    def visualize_components(self, subject_idx):
        original_file_name = os.path.join(self.original_dir, f'original_{subject_idx}.nii.gz')
        component_files = [
            os.path.join(self.processed_dir, f'processed_{subject_idx}_component_{i + 1}.nii.gz')
            for i in range(self.n_components)
        ]

        # Load the original MRI and find its intensity range
        original_nifti = nib.load(original_file_name)
        original_data = original_nifti.get_fdata()
        min_intensity = np.min(original_data)
        max_intensity = np.max(original_data)

        # Normalize each component image based on the original MRI intensity range
        for i, component_file in enumerate(component_files):
            component_nifti = nib.load(component_file)
            component_data = component_nifti.get_fdata()

            # Perform the intensity scaling
            normalized_component_data = (component_data - min_intensity) / (max_intensity - min_intensity)
            
            # Clip the values to be in [0,1] range after normalization
            normalized_component_data = np.clip(normalized_component_data, 0, 1)
            
            normalized_component_nifti = nib.Nifti1Image(normalized_component_data, component_nifti.affine)

            # Save the normalized component image back to the file
            nib.save(normalized_component_nifti, component_file)

        # Concatenate all the component file paths with the original MRI file path
        fsl_args = ['fsleyes', original_file_name] + component_files

        # Use FSLeyes to view the overlay components in a separate window
        subprocess.Popen(fsl_args).wait()  # Wait for the FSL window to be closed
    
    def process_all_subjects(self):
        start_time = time.time()
        for subject_idx, func_file in enumerate(self.func_filenames):
            print(f"Processing subject {subject_idx + 1}/{self.total_subjects}")
            self.process_subject(subject_idx, func_file)
            self.visualize_components(subject_idx)
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add positional arguments
    parser.add_argument("n_subjects", type=int, help="number of subjects")
    parser.add_argument("n_components", type=int, help="number of components")
    parser.add_argument("method", type=str, help="decomposition method: 'dict_learning' or 'ica'")

    # Add optional arguments
    parser.add_argument("--smoothing_fwhm", type=float, default=6.0, help="FWHM of Gaussian smoothing kernel")
    parser.add_argument("--memory", type=str, default="nilearn_cache", help="Directory to cache data")
    parser.add_argument("--memory_level", type=int, default=2, help="Level of memory caching")
    parser.add_argument("--random_state", type=int, default=0, help="Random state for reproducibility")
    parser.add_argument("--standardize", type=str, default="zscore_sample", help="Standardization method")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel")

    # Parse the arguments
    args = parser.parse_args()

    processor = NiftiViewer(args.n_subjects, args.n_components, args.method, args.smoothing_fwhm, args.memory, 
                            args.memory_level, args.random_state, args.standardize, args.n_jobs)
    processor.set_decomposition_object()
    processor.process_all_subjects()