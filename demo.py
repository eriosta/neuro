from clustering import *

# The threshold t in the extract_clusters method is a distance threshold that determines which components are grouped into the same cluster.
order_components = 20
correlation_tool = ComponentCorrelation(n_order=order_components)
correlation_tool.visualize_component_correlation()
clusters = correlation_tool.extract_clusters(t=1.5)
print(clusters)  # for debugging

# Fetch the ADHD200 resting-state fMRI dataset
n_subjects = 1
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
func_filenames = adhd_dataset.func

# Parameters
fwhm = 6

# Apply Dictionary Learning and visualize results for each cluster
for i, func_file in enumerate(func_filenames):
    for cluster_id, component_indices in clusters.items():
        print(f"Visualizing components for cluster {cluster_id}")
        visualizer = ComponentVisualization(func_file, order_components, component_indices, fwhm, i)  # Initialize with the correct order of arguments
        visualizer.process_and_visualize()
