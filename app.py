import os
import sys
import warnings
import nibabel as nib
from nilearn.datasets import fetch_adhd
from nilearn.decomposition import DictLearning, CanICA
import time
from joblib import Parallel, delayed
from tqdm import tqdm
import subprocess
import csv
import PySimpleGUI as sg
import argparse

# Ignore the FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


class NiftiProcessor:
    def __init__(self, n_subjects, n_components, save_nii_files):
        self.n_subjects = n_subjects
        self.n_components = n_components
        self.original_dir = './original_nii_files'
        self.processed_dir = './processed_nii_files'
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.func_filenames = fetch_adhd(n_subjects=self.n_subjects).func
        self.total_subjects = len(self.func_filenames)
        self.decomposition_object = None
        self.save_nii_files = save_nii_files

    def set_decomposition_object(self, decomposition_object):
        self.decomposition_object = decomposition_object

    def process_subject(self, subject_idx, func_file):
        self.decomposition_object.fit(func_file)

        # Save the original file
        original_file_name = os.path.join(self.original_dir, f'original_{subject_idx}.nii.gz')
        nib.save(nib.load(func_file), original_file_name)

        components_img = self.decomposition_object.components_img_

        # Generate a timestamp for the CSV file name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_file_name = f'{timestamp}.csv'

        # Open the CSV file for appending
        with open(csv_file_name, 'a', newline='') as f:
            writer = csv.writer(f)

            # Write the column names
            writer.writerow(['Subject', 'Component', 'Network', 'Laterality', 'Location'])

            # Iterate over each component and save it to a separate file
            for component_idx in range(components_img.shape[-1]):
                component_img = nib.Nifti1Image(
                    components_img.dataobj[..., component_idx], components_img.affine
                )
                component_file_name = os.path.join(
                    self.processed_dir, f'processed_{subject_idx}_component_{component_idx + 1}.nii.gz'
                )
                nib.save(component_img, component_file_name)

                # Open the original and the component image in FSLeyes
                fsl_process = subprocess.Popen(['fsleyes', original_file_name, component_file_name])

                # Create dropdown GUIs for choosing the component label, laterality, and location using PySimpleGUI
                layout = [
                    [sg.Text("Choose the network for the component:")],
                    [
                        sg.Combo(
                            [
                                'DEFAULT MODE',
                                'ATTENTION',
                                'EXECUTIVE CONTROL',
                                'SALIENCE',
                                'LANGUAGE',
                                'SENSORIMOTOR',
                                'VISUAL'
                            ],
                            key='label'
                        )
                    ],
                    [sg.Text("Choose the laterality:")],
                    [
                        sg.Combo(
                            ['Left', 'Right'],
                            key='laterality'
                        )
                    ],
                    [sg.Text("Choose the location:")],
                    [
                        sg.Combo(
                            ['Background', 'Frontal Pole', 'Insular Cortex', 'Superior Frontal Gyrus', 'Middle Frontal Gyrus', 'Inferior Frontal Gyrus, pars triangularis', 'Inferior Frontal Gyrus, pars opercularis', 'Precentral Gyrus', 'Temporal Pole', 'Superior Temporal Gyrus, anterior division', 'Superior Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, anterior division', 'Middle Temporal Gyrus, posterior division', 'Middle Temporal Gyrus, temporooccipital part', 'Inferior Temporal Gyrus, anterior division', 'Inferior Temporal Gyrus, posterior division', 'Inferior Temporal Gyrus, temporooccipital part', 'Postcentral Gyrus', 'Superior Parietal Lobule', 'Supramarginal Gyrus, anterior division', 'Supramarginal Gyrus, posterior division', 'Angular Gyrus', 'Lateral Occipital Cortex, superior division', 'Lateral Occipital Cortex, inferior division', 'Intracalcarine Cortex', 'Frontal Medial Cortex', 'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)', 'Subcallosal Cortex', 'Paracingulate Gyrus', 'Cingulate Gyrus, anterior division', 'Cingulate Gyrus, posterior division', 'Precuneous Cortex', 'Cuneal Cortex', 'Frontal Orbital Cortex', 'Parahippocampal Gyrus, anterior division', 'Parahippocampal Gyrus, posterior division', 'Lingual Gyrus', 'Temporal Fusiform Cortex, anterior division', 'Temporal Fusiform Cortex, posterior division', 'Temporal Occipital Fusiform Cortex', 'Occipital Fusiform Gyrus', 'Frontal Operculum Cortex', 'Central Opercular Cortex', 'Parietal Operculum Cortex', 'Planum Polare', "Heschl's Gyrus (includes H1 and H2)", 'Planum Temporale', 'Supracalcarine Cortex', 'Occipital Pole'],
                            key='location'
                        )
                    ],
                    [sg.Button('OK')]
                ]

                window = sg.Window('Network Selection', layout)

                while True:
                    event, values = window.read()

                    if event == 'OK':
                        label = values['label']
                        laterality = values['laterality']
                        location = values['location']
                        break

                window.close()

                # Write the chosen labels to the CSV file
                writer.writerow([subject_idx, component_idx + 1, label, laterality, location])

                # Close FSLeyes
                fsl_process.terminate()
                
                # Delete the original file if save_nii_files is False
                if not self.save_nii_files:
                    os.remove(original_file_name)
                    os.remove(component_file_name)

        return True


    def process_all_subjects(self):
        start_time = time.time()
        Parallel(n_jobs=-1)(delayed(self.process_subject)(subject_idx, func_file)
                            for subject_idx, func_file in tqdm(enumerate(self.func_filenames), total=self.total_subjects))
        print("--- %s seconds ---" % (time.time() - start_time))


class AlgorithmSelector:
    @staticmethod
    def select_algorithm():
        layout = [
            [sg.Text('Choose the algorithm:')],
            [sg.Radio('Dictionary Learning', "RADIO1", default=True, key='dict_learning'),
             sg.Radio('Independent Component Analysis (ICA)', "RADIO1", key='ica')],
            [sg.Button('OK')]
        ]

        window = sg.Window('Algorithm Selection', layout)

        while True:
            event, values = window.read()

            if event == 'OK':
                if values['dict_learning']:
                    algorithm = 'dict_learning'
                elif values['ica']:
                    algorithm = 'ica'
                break

        window.close()

        return algorithm


class DecompositionObject:
    def __init__(self, algorithm, n_components):
        if algorithm == 'dict_learning':
            self.decomposition_object = DictLearning(
                n_components=n_components,
                smoothing_fwhm=6.0,
                memory="nilearn_cache",
                memory_level=2,
                random_state=0,
                standardize="zscore_sample",
                n_jobs=-1,
            )
        elif algorithm == 'ica':
            self.decomposition_object = CanICA(
                n_components=n_components,
                smoothing_fwhm=6.0,
                memory="nilearn_cache",
                memory_level=2,
                random_state=0,
                standardize="zscore_sample",
                n_jobs=-1,
            )


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add positional arguments
    parser.add_argument("n_subjects", type=int, help="number of subjects")
    parser.add_argument("n_components", type=int, help="number of components")

    # Add optional arguments
    parser.add_argument("-s", "--save", action="store_true", help="save '.nii' files")

    # Parse the arguments
    args = parser.parse_args()

    processor = NiftiProcessor(args.n_subjects, args.n_components, args.save)
    algorithm = AlgorithmSelector.select_algorithm()
    decomposition_object = DecompositionObject(algorithm, args.n_components).decomposition_object

    processor.set_decomposition_object(decomposition_object)
    processor.process_all_subjects()