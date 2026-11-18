# -*- coding: utf-8 -*-
"""
This file contains a list of python script and their purpuse and status
"""
"""
0_environment.yml
This file includes a setup of the python environment requried to run the scripts.
Please note that installation of GPU resources are not added. The model wil 
automatically try to run the script on the GPU (if installed) and will otherwise
use the CPU.

"""

"""
1a_dataset_statistics.py
This is used to compute statistics of the dataset. 
1. Calculates Total Pixel Area (Resolution * Image Count).
2. Calculates "No Defect" (Background) pixel counts.
3. Calculates Pixel Percentages for all categories.
4. Maintains the TA, TB, TC split.
"""

"""
1b_histogram_plot.py
This script reads the segmented masks and plots histograms of the defect size 
distribution. It generates:
1. Individual plots for all datasets and individual plots for the tunnels 
TA, TB, TC.
2. A combined subplot figure comparing TA, TB, and TC.

The user chose which defect that should be plotted.
"""


"""
1c_create_classification.py
this scripts reads the csv files that contain information about images with and
withou cracks. Based on this, three classification datasets are created in the
folder "3_classification", i.e. TA, TB and TC. Each folder contains the 
subfolder "crack" and "no_crack"
"""

"""
2_train_CNN.py

This script trains a UNet segmentaiton model for a single detection class. 
The user defines the "Session_Name" which is the output folder for the saved 
model, plots and metrics. 
The user use the Global Configuration to adjust parameters. This includdes: 
 - A weight factor is included for imbalanced datasets.
- Data used for Traning, Evaluation and Testing is based on csv files.
 - The script creates masks used for the fastai packaage which use 1 for defect
 and 0 for background. The user defines the pixel value for the class they want 
to train the model for. 
 - Model training parameters are easily adjusted.
 - Output includes plots of top 5 best and worst predictions of cracks and 
 txt files with a summary of the metrics
"""

"""
2b_plot_training.py
This script reads the csv output from training and creates a plot of training 
and validation loss in one plot and IoU and F1-score in a second plot. 
User only needs to change "TRAINING_DATA" to correct training set.
"""

"""
3_evaluate_CNN.py
This script loads a pre-trained model and evaluate its performance on a list 
of datasets. The output is a .txt file with metrics. Naming of the file is based on
the SESSION_NAME and metrics for each eavluation is added in the txt file in
sequence, i.e. the metrics for all evaluation using the same model is stored in
the same file.
"""