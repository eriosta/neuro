import streamlit as st
import time  # Add the time module
import pandas as pd  # Add the pandas library
from clustering import *
from nilearn import datasets

order_components = 20
correlation_tool = ComponentCorrelation(n_order=order_components)

# Fetch the ADHD200 resting-state fMRI dataset
n_subjects = 1
adhd_dataset = datasets.fetch_adhd(n_subjects=n_subjects)
func_filenames = adhd_dataset.func
fwhm = 6

def main():
    st.title("Cluster Visualization Tool")
    
    # Create a slider to adjust t value
    t = st.slider("Set the distance threshold (t)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
    
    # Create a slider to adjust order_components value
    order_components = st.slider("Set the number of order_components", min_value=5, max_value=50, value=20, step=1)
    
    # Add "Run" button
    run_button = st.button("Run")
    
    if run_button:
        st.write(f"Visualizing component correlation with t = {t}")
        
        correlation_tool = ComponentCorrelation(n_order=order_components)  # Update correlation_tool with user-defined order_components
        correlation_tool.visualize_component_correlation(streamlit=True)
        clusters = correlation_tool.extract_clusters(t=t)
        
        # Convert clusters dictionary to a DataFrame
        clusters_df = pd.DataFrame([(cluster_id, component_indices) for cluster_id, component_indices in clusters.items()], columns=['Cluster', 'Component Indices'])
        clusters_df['Component Indices'] = clusters_df['Component Indices'].apply(lambda x: ', '.join(map(str, x)))
        
        # Display clusters in Streamlit with expandable sections
        st.write("Clusters:")
        for cluster_id, component_indices in clusters.items():
            with st.expander(f"Cluster {cluster_id}"):
                st.write("**Component Indices:**", ', '.join(map(str, component_indices)))

        
        # Display images corresponding to each cluster
        for i, func_file in enumerate(func_filenames):
            for cluster_id, component_indices in clusters.items():
                st.write(f"Visualizing components for cluster {cluster_id}")
                visualizer = ComponentVisualization(func_file, order_components, component_indices, fwhm, i)
                
                st.write("Processing and visualizing components...")
                
                start_time = time.time()  # Start measuring time
                visualizer.process_and_visualize(streamlit=True)
                end_time = time.time()  # Stop measuring time
                
                elapsed_time = end_time - start_time
                st.write(f"Time taken: {elapsed_time:.2f} seconds")
            
if __name__ == "__main__":
    main()
