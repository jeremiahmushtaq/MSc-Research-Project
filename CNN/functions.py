# Import dependancies
import os
import numpy as np

# Define function to load haploypes (training and testing)
def load_haplotypes(directory):
    # Set directory to access data
    os.chdir(directory)
    print(f"Current working directory: {os.getcwd()}")

    # Remove .DS_Store
    files = [f for f in os.listdir() if f != '.DS_Store']
    print(f"Number of files: {len(files)}")

    # Import haplotypes
    haplotypes = []
    for file in files:
        x = np.loadtxt(file, delimiter='\t', dtype=int)
        haplotypes.append(x)
    
    return haplotypes

# Define pre-processing function for testing data
def process_seqs(batch):
    # Instantize padding length error
    class PaddingError(Exception):
        pass
    # Function logic
    processed_seqs = []
    for sim in batch:
        temp = []
        for ind_seq in sim:
            if len(ind_seq) > 4882:
                x = ind_seq[:4882]
                temp.append(x)
            else:
                padding = 4882 - len(ind_seq)
                if padding < 0:
                    raise PaddingError("padding length is negative")
                else:
                    x = np.pad(array=ind_seq, pad_width=(0,padding), mode='constant', constant_values=-1)
                    temp.append(x)
        processed_seqs.append(temp)
    # Convert to numpy array
    return(np.array(processed_seqs))