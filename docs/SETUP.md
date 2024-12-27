# Setup

We assume the users are on Ubuntu machine and ready to have Python 3.10 installed. Then, the users can follow this instruction to install our RCAEval package. In addition, users who familiar with Continuous Integration (CI) can take a look at our CI configuration at [.circleci](.circleci) and [.github/workflows](.github/workflows)  to see how we install, test, and reproduce our RCAEval on different CI platforms including GitHub Action and CircleCI.

**Table of contents**

  * [Install Python and required packages](#install-python-and-required-packages)
  * [Clone RCAEval from GitHub](#clone-rcaeval-from-github)
  * [Install RCAEval](#install-rcaeval)
    + [Install RCAEval in DEFAULT mode](#install-rcaeval-in-default-mode)
    + [Install RCAEval in RCD mode](#install-rcaeval-in-rcd-mode)
    + [Install RCAEval in FGES mode](#install-rcaeval-in-fges-mode)
  * [Basic usage example](#basic-usage-example)



## Install Python and required packages

```bash
# install Python 3.8 and Python3.10
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update -y
sudo apt-get install -y python3.8 python3.8-dev python3.8-venv
sudo apt-get install -y python3.10 python3.10-dev python3.10-venv

# install required packages
sudo apt install -y build-essential \
  libxml2 libxml2-dev zlib1g-dev \
  python3-tk graphviz
```

## Clone RCAEval from GitHub

```bash
git clone https://github.com/phamquiluan/RCAEval.git && cd RCAEval
```

## Install RCAEval

We maintain three difference modes: `default`, `rcd`, `fges` for RCAEval due to the dependency constraints. Specifically, most methods use the `default` mode. RCD method uses the `rcd` mode. The fGES-based methods uses the `fges` mode. We can easily install RCAEval in different modes as follows.


### Install RCAEval in DEFAULT mode

**Create and activate a virtual environment**

```bash
# create a virtual environment
python3.10 -m venv env

# activate the environment
. env/bin/activate
```

**Install RCAEval using Pip**

```bash
# build RCAEval from source
pip install pip==20.0.2
pip install -e .[default]
```

**Test the installation**

Users can perform testing using the following commands:

```bash
pytest tests/test.py
```

<details>
<summary>The expected output would look like this</summary>

```bash
$ pytest tests/test.py 
============================== test session starts ===============================
platform linux -- Python 3.10.12, pytest-7.3.1, pluggy-1.0.0
rootdir: /home/ubuntu/RCAEval
plugins: dvc-2.57.3, hydra-core-1.3.2
collected 11 items                                                               

tests/test.py ...........                                                  [100%]

========================= 11 passed in 135.27s (0:02:15) =========================
```
</details>


### Install RCAEval in RCD mode

**Create and activate a virtual environment**

```bash
# create a virtual environment
python3.8 -m venv env-rcd

# activate the environment
. env-rcd/bin/activate
```

**Install using Pip**

```bash
pip install pip==20.0.2
pip install wheel
pip install -e .[rcd]

#IMPORTANT, run the following command to link the customized PC
bash script/link.sh
```

**Reproduce RCD**

Users can reproduce the RCA performance of the RCD method using the following command:

```bash
python main.py --method rcd --dataset online-boutique
```

<details>
<summary>The expected output would look like this (it takes ~33 minutes for 1 iteration)</summary>

```bash
$ python main.py --method rcd --dataset online-boutique
100%|███████████████████████████████████████████| 125/125 [33:44<00:00, 16.20s/it]
--- Evaluation results ---
Avg@5-CPU:   0.94
Avg@5-MEM:   0.67
Avg@5-DISK:  0.68
Avg@5-DELAY: 0.25
Avg@5-LOSS:  0.51
---
Avg speed: 16.2
```
</details>

### Install RCAEval in fGES mode

**Run the following commands**

```bash
python3.8 -m venv env-fges
. env-fges/bin/activate

pip install pip==20.0.2
pip install -e .[fges]

cd LIB
pip install -e .
cd ..

# it MUST be performed in this order
sudo apt-get install -y gcc graphviz libgraphviz-dev pkg-config
pip install dill pygobnilp
pip install -U numba
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install category_encoders sortedcontainers fcit
pip install pgmpy
pip install feature_engine
```

**Test the installation**

Users can perform testing using the following commands:

```bash
python main.py --method fges_randomwalk --dataset online-boutique --test
```

<details>
<summary>The expected output would look like this</summary>

```bash
$ python main.py --method fges_randomwalk --dataset online-boutique --test
100%|███████████████████████████████████████████████| 2/2 [01:04<00:00, 32.33s/it]
--- Evaluation results ---
Avg@5-CPU:   0.94
Avg@5-MEM:   0.67
Avg@5-DISK:  0.7
Avg@5-DELAY: 0.25
Avg@5-LOSS:  0.51
---
Avg speed: 32.33
```
</details>

## Basic usage example

Users can check the basic usage example of RCAEval in the [README.md#basic-usage-example](README.md#how-to-use) section.
