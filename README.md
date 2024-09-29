# Hybrid Quantum Neural Network-based User Localization

This repository contains the implementation and datasets for the paper **"Hybrid Quantum Neural Network-based Indoor User Localization using Cloud Quantum Computing."** The project leverages a hybrid quantum-classical approach to predict user locations based on environmental data.

## Repository Structure

The repository is structured into 3 main folders and a Jupyter notebook file containing the code for the implementation.

### 1. `Scenario1/`
This folder contains the dataset for **Scenario 1**, representing the first environment setup where data was collected for indoor user localization. The files in this folder include the input features and the corresponding labels for this specific scenario.

### 2. `Scenario 2/`
This folder contains the dataset for **Scenario 2**, representing a different environmental setup. The data format is consistent with Scenario 1, but the environmental variables differ to simulate varying conditions for user localization.

### 3. `Scenario3/`
This folder contains the dataset for **Scenario 3**, representing another unique environment for indoor localization. Similar to the other two scenarios, the dataset in this folder captures different factors influencing user location predictions.

### 4. `HQNN_user_localization.ipynb`
The Jupyter notebook contains the implementation of the **Hybrid Quantum Neural Network** used for predicting user locations based on the datasets provided. The notebook includes:
- **Data Preprocessing:** Steps to prepare the datasets for training and testing.
- **Quantum Layer Construction:** Implementation of the quantum circuits used in the neural network.
- **Classical Neural Network Layers:** Integration of classical layers in a hybrid approach.
- **Training and Evaluation:** Code for training the hybrid model and evaluating its performance across the three scenarios.
