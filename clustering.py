import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from nilearn import datasets, image, plotting, decomposition
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.cluster.hierarchy import fcluster
import streamlit as st

class ComponentCorrelation:
    def __init__(self, n_order, memory_level=2, cache_dir="nilearn_cache"):
        self.n_order = n_order
        self.cache_dir = cache_dir
        self.memory_level = memory_level

    def _fetch_data(self):
        """Fetch sample functional data for testing."""
        dataset = datasets.fetch_adhd(n_subjects=1)
        self.func_filename = [image.concat_imgs(dataset.func)]
        self.affine = self.func_filename[0].affine

    def _perform_decomposition(self):
        options = {
            "random_state": 0,
            "memory": self.cache_dir,
            "memory_level": self.memory_level
        }
        dict_learn = decomposition.DictLearning(n_components=self.n_order, **options)
        results = dict_learn.fit_transform(self.func_filename)
        self.components_img = results[0]

    def _compute_correlation_matrix(self):
        self.correlation_matrix = np.zeros((self.n_order, self.n_order))
        self.results = []
        for i in range(self.n_order):
            for j in range(self.n_order):
                data_i = self.components_img[..., i]
                data_j = self.components_img[..., j]
                if data_i.size > 1 and data_j.size > 1:
                    correlation, p_value = pearsonr(data_i.ravel(), data_j.ravel())
                    self.results.append({
                        'Component_1': i,
                        'Component_2': j,
                        'Pearson_r': correlation,
                        'p_value': p_value
                    })
                    self.correlation_matrix[i, j] = correlation
        self.correlation_matrix = np.nan_to_num(self.correlation_matrix)

    def _plot_heatmap(self,streamlit=None):
        diverging_cmap = plt.cm.RdBu_r
        figsize = (10, 5)
        cluster_grid = sns.clustermap(self.correlation_matrix, method="average", cmap=diverging_cmap, vmin=-1, vmax=1, annot=False, fmt=".2f", figsize=figsize)
        plt.show()
        if streamlit is not None:
            st.pyplot(plt)
        
        # Get the order of the components after hierarchical clustering
        self.ordered_components = leaves_list(linkage(self.correlation_matrix, method='average'))

    def get_ordered_components(self):
        """Return the ordered list of component indices."""
        if hasattr(self, 'ordered_components'):
            return self.ordered_components
        else:
            raise AttributeError("Please run 'visualize_component_correlation' first to generate the ordered components.")

    def export_results_to_csv(self, filename="correlation_results.csv"):
        df = pd.DataFrame(self.results)
        df = df.sort_values(by='p_value')
        df.to_csv(filename, index=False)

    def visualize_component_correlation(self,streamlit):
        self._fetch_data()
        self._perform_decomposition()
        self._compute_correlation_matrix()
        self._plot_heatmap(streamlit)
        self.export_results_to_csv()
        
    def extract_clusters(self, t=1.5):
        """
        Extract clusters from the correlation matrix using hierarchical clustering.
        
        Parameters:
            t: float, optional (default=1.5)
                The threshold to form clusters. Components that are closer 
                than this threshold in the dendrogram will belong to the same cluster.
        
        Returns:
            dict
                A dictionary with cluster identifiers as keys and lists of 
                component indices as values.
        """
        Z = linkage(self.correlation_matrix, method='average')
        cluster_assignments = fcluster(Z, t, criterion='distance')
        clusters = {}
        for idx, cluster_id in enumerate(cluster_assignments):
            clusters.setdefault(cluster_id, []).append(idx)
        return clusters


class ComponentVisualization:
    def __init__(self, func_file,n_components,component_indices, fwhm, subject_index, ordered_components=None):
        self.func_file = func_file
        self.component_indices = component_indices
        self.fwhm = fwhm
        self.subject_index = subject_index
        self.n_components = n_components
        self.bg_img = datasets.load_mni152_template()
        if ordered_components is not None:
            self.ordered_components = ordered_components
        else:
            self.ordered_components = component_indices

    def apply_dictionary_learning(self):
        fmri_subject = image.smooth_img(self.func_file, self.fwhm)
        dict_learn_subject = decomposition.DictLearning(n_components=self.n_components, random_state=0)
        dict_learn_subject.fit(fmri_subject)
        self.components_img_subject = dict_learn_subject.components_img_

    def visualize_components(self,streamlit=None):
        
        n_cols = len(self.component_indices)  # Determine number of columns by the length of the list of component indices
        n_rows = 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15 * n_cols, 15))
        
        # If there's only one component, make sure axes is an array for consistency
        if n_cols == 1:
            axes = np.array([axes])
        
        for idx, component in enumerate(self.component_indices):
            ax = axes[idx]
            component_img = image.index_img(self.components_img_subject, component)
            y_coord = plotting.find_xyz_cut_coords(component_img)[1]
            title_component = f'S{self.subject_index}C{component}'
            plotting.plot_stat_map(component_img, bg_img=self.bg_img, cut_coords=[y_coord], display_mode='y', title=title_component, axes=ax, colorbar=False)
        plt.tight_layout()
        plt.show()
        
        if streamlit is not None:
            st.pyplot(plt)
            
    def process_and_visualize(self,streamlit):
        self.apply_dictionary_learning()
        self.visualize_components(streamlit)

