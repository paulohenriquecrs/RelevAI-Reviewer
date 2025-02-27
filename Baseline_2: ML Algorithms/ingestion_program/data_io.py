""" The `Data_io` file comprises fundamental methods primarily designed 
    for tasks related to data reading, data loading, data writing, and
    displaying data.
 """

# -------------------------------------
# Imports
# -------------------------------------
import os
import numpy as np
import pandas as pd
from zipfile import ZipFile, ZIP_DEFLATED
from contextlib import closing
import json


# -------------------------------------
# Load Data
# -------------------------------------
def load_data(_data_dir, data_type="train", index=False):
    """
    The method Load data from specified directories.

    Parameters:
        _data_dir (str): Base directory containing 'train' or 'test' subdirectories.
        data_type (str): Type of data to load ('train' or 'test').
        index (bool): Flag to determine if index should be used.

    Returns:
        list: List of dictionaries containing data, labels, and settings.

    Example:
        load_data("/path/to/data", data_type="train", index=False)
    """

    print("\n\n###-------------------------------------###")
    print("### Data Loading")
    print("###-------------------------------------###\n")

    # set train and test directories
    data_dir = os.path.join(_data_dir, data_type, "data")
    labels_dir = os.path.join(_data_dir, data_type, "labels")
    settings_dir = os.path.join(_data_dir, data_type, "settings")

    # print directories
    print("[*] data dir : ", data_dir)
    print("[*] labels dir : ", labels_dir)
    print("[*] settings dir : ", settings_dir)

    # check if directories exist
    if not os.path.exists(data_dir):
        print("[-] data dir : ", data_dir, " not found")
        return
    else:
        print("[+] data dir found")

    if not os.path.exists(labels_dir):
        print("[-] labels dir : ", labels_dir, " not found")
        return
    else:
        print("[+] labels dir found")

    if not os.path.exists(settings_dir):
        print("[-] settings dir : ", settings_dir, " not found")
        return
    else:
        print("[+] settings dir found")

    # train and test files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.labels')]
    settings_files = [f for f in os.listdir(settings_dir) if f.endswith('.json')]

    # check if files exist
    if len(settings_dir) != len(labels_files) != len(settings_files):
        print("[-] Number of data, labels, and settings files do not match! ")
        return

    total_files = len(data_files)

    print("[+] {} datsets found".format(total_files))

    datasets = []

    if not index and total_files == 1:

        data_file = "data.csv"
        labels_file = "data.labels"
        settings_file = "settings.json"

        data_file_path = os.path.join(data_dir, data_file)
        labels_file_path = os.path.join(labels_dir, labels_file)
        settings_file_path = os.path.join(settings_dir, settings_file)

        datasets.append({
            "data": read_data_file(data_file_path),
            "labels": read_labels_file(labels_file_path),
            "settings": read_json_file(settings_file_path)
        })

    else:
        for i in range(0, total_files):

            data_file = "data_"+str(i+1)+".csv"
            labels_file = "data_"+str(i+1)+".labels"
            settings_file = "settings_"+str(i+1)+".json"

            data_file_path = os.path.join(data_dir, data_file)
            labels_file_path = os.path.join(labels_dir, labels_file)
            settings_file_path = os.path.join(settings_dir, settings_file)

            datasets.append({
                "data": read_data_file(data_file_path),
                "labels": read_labels_file(labels_file_path),
                "settings": read_json_file(settings_file_path)
            })

    print("---------------------------------")
    print("[+] Data loaded!")
    print("---------------------------------\n\n")
    return datasets


# -------------------------------------
# Read Data File
# ------------------------------------
def read_data_file(data_file):
    """
    Read a CSV data file.

    Parameters:
        data_file (str): Path to the CSV data file.

    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame.

    Example:
        read_data_file("/path/to/data.csv")
    """

    # check data file
    if not os.path.isfile(data_file):
        print("[-] data file {} does not exist".format(data_file))
        return

    # Load data file
    df = pd.read_csv(data_file)

    return df


# -------------------------------------
# Read Labels File
# -------------------------------------
def read_labels_file(labels_file):
    """
    Read labels from a labels file.

    Parameters:
        labels_file (str): Path to the labels file.

    Returns:
        np.ndarray: Loaded labels as a NumPy array.

    Example:
        read_labels_file("/path/to/labels.labels")
    """

    # check labels file
    if not os.path.isfile(labels_file):
        print("[-] labels file {} does not exist".format(labels_file))
        return

    # Read labels file
    f = open(labels_file, "r")
    labels = f.read().splitlines()
    labels = np.array(labels, dtype=float)
    f.close()
    return labels


# -------------------------------------
# Read Json File
# -------------------------------------
def read_json_file(json_file):
    """
    Read data from a JSON file.

    Parameters:
        json_file (str): Path to the JSON file.

    Returns:
        dict: Loaded data as a dictionary.

    Example:
        read_json_file("/path/to/settings.json")
    """

    # check json file
    if not os.path.isfile(json_file):
        print("[-] json file {} does not exist".format(json_file))
        return
    return json.load(open(json_file))


# -------------------------------------
# Data Statistics
# -------------------------------------
def show_data_statistics(data_sets, name="Train"):
    """
    Display statistics about the loaded datasets.

    Parameters:
        data_sets (list): List of datasets.
        name (str): Name of the datasets (default is "Train").

    Example:
        show_data_statistics(data_sets, name="Test")
    """

    print("###-------------------------------------###")
    print("### Data Statistics " + name)
    print("###-------------------------------------###")

    for index, data_set in enumerate(data_sets):
        print("-------------------")
        print("Set " + str(index+1))
        print("-------------------")

        print("[*] Total points: ", data_set["data"].shape[0])
        if "labels" in data_set:
            print("[*] Background points: ", len(data_set["labels"]) - np.count_nonzero(data_set["labels"] == 1))
            print("[*] Signal points: ", np.count_nonzero(data_set["labels"] == 1))


# -------------------------------------
# Write Predictions
# -------------------------------------
def write(filename, predictions):
    """
    Write predictions to a file.

    Parameters:
        filename (str): Path to the output file.
        predictions (list): List of predictions.

    Example:
        write("/path/to/predictions.txt", predictions)
    """

    with open(filename, 'w') as f:
        for ind, lbl in enumerate(predictions):
            str_label = str(float(lbl))
            if ind < len(predictions)-1:
                f.write(str_label + "\n")
            else:
                f.write(str_label)


# -------------------------------------
# Zip folder
# -------------------------------------

def zipdir(archivename, basedir):
    """
    Zip a directory, excluding the '__pycache__' directory.

    Parameters:
        archivename (str): Name of the output zip file.
        basedir (str): Path to the directory to be zipped.

    Example:
        zipdir("name.zip", "/path/to/source/dir")
    """
    '''Zip directory, from J.F. Sebastian http://stackoverflow.com/'''
    assert os.path.isdir(basedir)
    with closing(ZipFile(archivename, "w", ZIP_DEFLATED)) as z:
        for root, dirs, files in os.walk(basedir):
            # Exclude __pycache__ directory
            dirs[:] = [d for d in dirs if d != '__pycache__']

            # NOTE: ignore empty directories
            for fn in files:
                if fn[-4:] != '.zip' and fn != '.DS_Store':
                    absfn = os.path.join(root, fn)
                    zfn = absfn[len(basedir):]  # XXX: relative path
                    z.write(absfn, zfn)
