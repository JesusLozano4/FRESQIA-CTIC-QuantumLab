# Foliar REcognition of Symptoms using Quantum and Artificial Intelligence: FRESQIA
This project originates from previous work on the **ARQA Demonstrator 1**, within the framework of **[Cervera Network ARQA](https://redarqa.es/)**. The architectures and structures developed in that phase have been adapted and repurposed to create a disease detection system for strawberry leaves. This initiative aims to increase the visibility of **Quantum Computing** within the agricultural sector, showcasing its practical utility and its promise for future scalability.

## Project Structure
- dataset
  - hojas2clases: Unzip the compressed file inside here
  - HojasSimulador: Unzip the compressed file inside here
- notebooks
  - CropLeaves.ipynb
  - HQNN.ipynb
  - QCNN.ipynb
  - TrainHQNN.ipynb
  - TrainQCNN.ipynb
- src
  - nn
    - ansatz
    - encodings
    - measurements
    - models
      - hybrid
        - HQNN_Parallel.py
        - HQNN_quanv.py
      - quantum
        - QCNN.py
    - qlayers
      - quantum_linear.py
      - quanvolution.py
  - utils
    - dataset.py
    - plotting.py
    - reshape_data.py
    - run_conf.py
    - run_experiment.py
    - run_inference.py
    - training.py
  - YOLO
    - yolo_funcs.py
- requirements.txt

## Installation
To install the required libraries and dependencies, run the following command in your terminal:

`pip install -r requirements.txt`

## Setup and Data (Releases)
The dataset and trained models are hosted in the **Releases** section of this repository. Please download the files and organize the `dataset` directory as follows:

- dataset/
  - 2hojas_HybridCNN.pth
  - 2hojas_QCNN.pth
  - best.pt
  - hojas2clases/
    - Data1/ (Unzip images here)
  - HojasSimulador/
    - Data2/ (Unzip images here)

## Usage Workflow
The correct workflow for using this project is as follows:

1. **Preprocessing:** First, create the leaf crops by running the `CropLeaves.ipynb` notebook.
2. **Model Testing:** Depending on whether you want to test the Hybrid or Quantum model, use `HQNN.ipynb` or `QCNN.ipynb` respectively.
3. **Training & Hyperparameters:** If you wish to modify the training hyperparameters for the classification models, use `TrainHQNN.ipynb` or `TrainQCNN.ipynb`.
