# main.py

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from generate_label import generate_label
from spxy_image import spxy_image
from spxy import spxy

def load_hyperspectral_data(file_path):
    data = loadmat(file_path)
    return data['data'], data['target']

def main():
    current_file_path = os.path.abspath("__file__")  
    print(current_file_path)

    # Assuming that the hyperspectral image data is stored in Indian_pines_corrected.mat
    # and the corresponding ground truth labels are stored in Indian_pines_gt.mat

    # Load hyperspectral image data using scipy.io
    hyperspectral_data, gt_labels = load_hyperspectral_data(os.path.join('data', 'Indian_pines_corrected.mat'))

    # Flatten the data
    x = hyperspectral_data.reshape((-1, hyperspectral_data.shape[-1]))

    # Generate indices for spxy function
    indexx = np.arange(1, len(gt_labels) + 1)

    # Use spxy algorithm to split the dataset
    spec_train, spec_test, target_train, target_test, index_train, index_test = spxy(x, gt_labels, indexx, test_size=0.2)

    # Save original data to CSV
    original_data = pd.DataFrame(x, columns=[f'Band_{i}' for i in range(1, x.shape[1] + 1)])
    original_data['GT_Labels'] = gt_labels
    original_data['Index'] = indexx
    original_data.to_csv(current_file_path.split('src')[0] + '\\original_data.csv', index=False)

    # Save processed data to CSV
    processed_data_train = pd.DataFrame(spec_train, columns=[f'Band_{i}' for i in range(1, spec_train.shape[1] + 1)])
    processed_data_train['GT_Labels'] = target_train
    processed_data_train['Index'] = index_train
    processed_data_train.to_csv(current_file_path.split('src')[0] + '\\processed_data_train.csv', index=False)

    processed_data_test = pd.DataFrame(spec_test, columns=[f'Band_{i}' for i in range(1, spec_test.shape[1] + 1)])
    processed_data_test['GT_Labels'] = target_test
    processed_data_test['Index'] = index_test
    processed_data_test.to_csv(current_file_path.split('src')[0] + '\\processed_data_test.csv', index=False)

    # Using spxy_image function to split the image data
    spxy_image('data', 'spxy_data')

    # Generate labels for the image data
    generate_label('spxy_data\\train', 'train')
    generate_label('spxy_data\\val', 'val')

    print("Processing completed.")

if __name__ == '__main__':
    main()
