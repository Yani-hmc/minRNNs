# Were RNNs All We Needed?

**Gabriel Mariadass - Yani Hammache**  
Deep Learning - Master MVA  
January 2026

---

## Origin of the Code

Part of this repository is based on the **official implementation** of the paper:

**Were RNNs All We Needed?**  
https://arxiv.org/abs/2410.01201  

Original repository:  
https://github.com/rbcborealis/minRNNs

The original repository provides the implementation of the **parallelized minGRU and minLSTM architectures** as well as the training and evaluation pipeline used in the paper.

This project reproduces and extends some of the experiments in the context of the **Deep Learning course of the MVA Master**.

---

## License

The original project is released under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.

Accordingly:
- The original authors are credited.
- The code is used **for non-commercial academic purposes**.
- Any derivative work must be distributed under the **same license**.

License:  
https://creativecommons.org/licenses/by-nc-sa/4.0/

---

## Repository Structure

- `main.py` contains the main training and evaluation pipeline.  
- `models.py` defines the different model architectures used in the experiments.  
- `minRNNs.py` implements the **minGRU** and **minLSTM** architectures.  
- `utils.py` contains utility functions used throughout the project.  
- `baselines.py` implements baseline RNN models used for comparison.  
- `configs/` contains configuration files for the different tasks.  
- `data/` contains the generation code for the synthetic sequence tasks.  
- `results/` stores trained models and experiment logs.  
- `eval_datasets/` stores generated evaluation datasets.  
- `depth_abl_test.py` implements experiments studying the effect of model depth.  
- `length_gen_test.py` implements experiments on sequence length generalization.  
- `Project.ipynb` is the notebook used to run and organize experiments.
