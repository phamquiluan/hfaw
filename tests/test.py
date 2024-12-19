"""Tests."""
import os
import shutil
from os import path
import subprocess
import pytest

from typing import Callable
import numpy as np
import pandas as pd
import pytest
import tempfile

from RCAEval.utility import (
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset,
)


@pytest.mark.parametrize("func", [
    download_re1_dataset,
    download_re2_dataset,
    download_re3_dataset,
])
def test_download_dataset(func: Callable):
    """Test download dataset."""
    local_path = tempfile.NamedTemporaryFile().name
    func(local_path=local_path)
    assert path.exists(local_path), local_path
    shutil.rmtree(local_path)    
    

def test_rq2_script():
    command = ["python", "rq2.py", "--method", "baro", "--dataset", "online-boutique", "--test"]
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the script ran successfully
    assert result.returncode == 0, f"Script failed with return code {result.returncode}\nOutput: {result.stdout}\nError: {result.stderr}"


def test_basic():
    # You can put the code here to a file named test.py
    from RCAEval.e2e import baro
    from RCAEval.utility import download_data, read_data

    # download a sample data to data.csv
    download_data()

    # read data from data.csv
    data = read_data("data.csv")
    anomaly_detected_timestamp = 1692569339

    # perform root cause analysis
    root_causes = baro(data, anomaly_detected_timestamp)["ranks"]

    # print the top 5 root causes
    print("Top 5 root causes:", root_causes[:5])
