# 🕵️ Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?

[![DOI](https://zenodo.org/badge/840137303.svg)](https://zenodo.org/doi/10.5281/zenodo.13294048)
[![pypi package](https://img.shields.io/pypi/v/RCAEval.svg)](https://pypi.org/project/RCAEval)
[![Build and test](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/build-and-test.yml)
[![Upload Python Package](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml/badge.svg)](https://github.com/phamquiluan/RCAEval/actions/workflows/python-publish.yml)

This repository includes artifacts for reuse and reproduction of experimental results presented in our ASE'24 paper titled "Root Cause Analysis for Microservices based on Causal Inference: How Far Are We?".

**Table of Contents** 
  * [Installation](#installation)
  * [How-to-use](#how-to-use)
    + [Data format](#data-format)
    + [Basic usage example](#basic-usage-example)
  * [Reproducibility](#reproducibility)
    + [Reproduce RQ1 - Causal Discovery Performance](#reproduce-rq1---causal-discovery-performance)
    + [Reproduce RQ2 - Root Cause Analysis Performance](#reproduce-rq2---root-cause-analysis-performance)
    + [Reproduce RQ3 - Efficiency](#reproduce-rq3---efficiency)
    + [Reproduce RQ4 - Data lengths](#reproduce-rq4---data-lengths)
  * [Download Experimental Data - Supplementary Material](#download-experimental-data---supplementary-material)
  * [Licensing](#licensing)
  * [Acknowledgments](#acknowledgments)


## Installation

We maintain 3 separate environments (`default`, `rcd` and `fges`) due to the dependency constraints of certain methods. Detailed installation instructions for all environments can be found in [INSTALL.md](INSTALL.md).

The `default` environment, are used for most methods, can be easily installed as follows.


**Open your terminal and run the following commands**

```bash
sudo apt update -y
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

**Clone RCAEval from GitHub**

```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```

**Install RCAEval from PyPI**

```bash
# Install RCAEval from PyPI
pip install RCAEval[default]
```

**OR, build RCAEval from source**

```bash
pip install pip==20.0.2
pip install -e .[default]
```


## How-to-use

### Data format

TBD

### Basic usage example

TBD


## Reproducibility

### Reproduce RQ1 - Causal Discovery Performance

We have provided a file named `graph_eval.py` to assist in reproducing the causal discovery results of our paper, which can be run using Python with the following syntax: 

```
python graph_eval.py [-h] [-i INPUT_PATH] [-o OUTPUT_PATH] [--length LENGTH] [-m MODEL]
```
The description for the arguments/options of the file `graph_eval.py` are as follows:

```
options:
  -h, --help            show this help message and exit
  --input-path          Choose a dataset to analyse. Options:
                        ['syn']
                        path to data
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        for results and reports
  --length LENGTH       length of time series
  -m MODEL, --model MODEL
                        func name
```



To reproduce the causal discovery performance, as presented in Table 3. You can download the corresponding dataset and extracted to folder `./data`. Then, you can run the file `graph_eval.py` to obtain the results for one iteration. For example:

As presented in Table 3, PC achieves F1, F1-S, and SHD of 0.49, 0.65, and 16 on the CIRCA 10 dataset. To reproduce this results as presented in the Table 3. You can run the following commands:

```bash
python graph_eval.py -i data/syn_circa/10 -m pc
```

The expected output should be exactly as presented in the paper (it takes around 1 minute to run the code)

```
F1:   0.49
F1-S: 0.65
SHD:  16
```

We can replace the method `pc` by methods `fci`, `granger`, `ICALiNGAM`, `DirectLiNGAM`, `ges`, `pcmci`,  etc. to replicate their results on the CIRCA 10 dataset present in Table 3.
We can also replace the dataset `syn_circa/10` by `syn_rcd/10`, 





### Reproduce RQ2 - Root Cause Analysis Performance

To reproduce the root cause analysis performance, as presented in Table 5. You can download the corresponding dataset and extracted to folder `./data`. Then, you can run the file `eval.py` to reproduce the results. For example:


As presented in Table 5, NSigma [ $t_\Delta = 0$ ] achieves Avg@5 of 0.94, 1, 0.9, 0.98, and 0.67 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce the RCA performance of NSigma [ $t_\Delta = 0$ ] as presented in the Table 5. You can run the following commands:

```bash
python eval.py -i data/online-boutique -o output-tmp -m nsigma --iter-num 10 -w 10 --length 10
```

<details>
<summary>Expected output after running the above code (it takes around 1 minute)</summary>

<br />

The results are exactly as presented in the paper (Table 5).
```
Evaluation results
s_cpu: 0.94
s_mem: 1.0
s_disk: 0.9
s_delay: 0.98
s_loss: 0.67
```
</details>

As presented in Table 5, NSigma [ $t_\Delta = 60$ ] achieves Avg@5 of 0.16, 0.24, 0.43, 0.55, and 0.38 for CPU, MEM, DISK, DELAY, and LOSS fault types on the Online Boutique dataset. To reproduce the RCA performance of NSigma [ $t_\Delta = 60$ ] as presented in the Table 5. You can run the following commands:

```bash
python eval.py -i data/online-boutique -o output-tmp -m nsigma --iter-num 10 -w 10 --length 10 --ad-delay 60
```

<details>
<summary>Expected output after running the above code (it takes around 1 minute)</summary>

<br />

The results are exactly as presented in the paper (Table 5).
```
Evaluation results
s_cpu: 0.16
s_mem: 0.24
s_disk: 0.43
s_delay: 0.55
s_loss: 0.38
```
</details>


We can replace the method `nsigma` by `baro`, `pc_pagerank`, `fci_pagerank`, `rcd`, `e_diagnosis`, etc. to replicate corresponding results.

### Reproduce RQ3 - Efficiency

The efficiency is captured in our evaluation script and saved in the corresponding output directory.

### Reproduce RQ4 - Data lengths

Our RQ4 relies on the scripts of RQ1 and RQ2, which we presented above.

## Download Experimental Data - Supplementary Material

Our datasets and their description are publicly available in Zenodo repository with the following information:
- Dataset DOI: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13305663.svg)](https://doi.org/10.5281/zenodo.13305663)
- Dataset URL: https://zenodo.org/records/13305663

In addition, you can download our supplementary material and all the data we used for this research from this [Google Drive](https://drive.google.com/drive/folders/1BG2P1ETEyKW62dU0I1ZpE64Ng9fy5_ju?usp=sharing).

## Licensing

This repository includes code from various sources with different licenses:

- **CIRCA**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_CIRCA). Original source: [CIRCA GitHub Repository](https://github.com/NetManAIOps/CIRCA).
- **RCD**: Licensed under the [MIT License](LICENSES/LICENSE_RCD). Original source: [RCD GitHub Repository](https://github.com/azamikram/rcd).
- **E-Diagnosis**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_E-Diagnosis). Original source: [PyRCA GitHub Repository](https://github.com/salesforce/PyRCA).
- **CausalAI**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_CausalAI). Original source: [CausalAI GitHub Repository](https://github.com/salesforce/causalai).
- **MicroCause**: Licensed under the [BSD 3-Clause License](LICENSES/LICENSE_MicroCause). Original source: [MicroCause GitHub Repository](https://github.com/PanYicheng/dycause_rca).

We have included their corresponding LICENSE into the [LICENSES](LICENSES) directory. For the code implemented by us, we distribute them under the [MIT LICENSE](LICENSE).

## Acknowledgments

We would like to express our sincere gratitude to the researchers and developers who created the baselines used in our study. Their work has been instrumental in making this project possible. We deeply appreciate the time, effort, and expertise that have gone into developing and maintaining these resources. Your commitment to advancing the field of root cause analysis and making your work accessible to the research community has been invaluable. This project would not have been feasible without your contributions.
