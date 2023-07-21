import nilearn.datasets as datasets

# Load the Harvard-Oxford atlas
harvard_oxford = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')

# Print the region/structure labels
print(harvard_oxford['labels'])
