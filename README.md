# PyRCN: A Toolbox for Exploration and Application of Reservoir Computing Networks
## Metadata
- Author: [Peter Steiner](mailto:peter.steiner@tu-dresden.de), Azarakhsh Jalalvand, 
Simon Stone and Peter Birkholz
- Journal: Engineering Applications of Artificial Intelligence, International 
Federation of Automatic Control, Elsevier.
- Weblink: [https://arxiv.org/abs/2103.04807](https://arxiv.org/abs/2103.04807)

## Summary and Contents
This repository contains supplemental material for the research paper entitled "PyRCN:
 A toolbox for Exploration and Application of Reservoir Computing Networks".

PyRCN is a toolbox for Reservoir Computing Networks (RCNs), which belong to a group of
machine learning techniques that project the input space non-linearly into a 
high-dimensional feature spaace. Since we introduce PyRCN, this repository contains 
all code examples and the entire benchmark test to compare PyRCN with other toolboxes.

## File list
- The following scripts are provided in this repository
    - `scripts/create_venv.sh`: UNIX Bash script to set up a virtual environment with 
    all required packages.
    - `scripts/run.sh`: UNIX Bash script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.sh`: UNIX Bash script to start the Jupyter Server for the 
    experiments.
    - `scripts/create_venv.ps1`: Windows PowerShell script to set up a virtual 
    environment with all required packages.
    - `scripts/run.ps1`: Windows PowerShell script to reproduce the results in the paper.
    - `scripts/run_jupyter-lab.ps1`: Windows PowerShell script to start the Jupyter Server for the 
    experiments. 
- The following python code is provided in `src`
    - `pyESN/`: The [pyESN](https://github.com/cknd/pyESN) class by 
    [cknd](https://github.com/cknd/)
    - `src/adapter.py`: Adapter classes to make other toolboxes sklearn-compatible.
    - `src/arima.py`: Wrapper functions for ARIMA.
    - `src/file_handling.py`: Functions to export results as CSV files.
    - `src/model_selection.py`: Wrapper class around 
    `sklearn.model_selection.PredefinedSplit` to support splitting a dataset in 
    training/validation/test.
    - `src/preprocessing.py`: Utility functions for preprocessing the dataset.
    - `src/main.py`: The main script to reproduce all results for stock price 
    volatility prediction.
    - `src/PyRCN-Intro.ipynb`: The Jupyter-Notebook containing the examples how to set
     up RCNs using PyRCN and its included building blocks.
- `requirements.txt`: Text file containing all required Python modules to be installed.
- `README.md`: The README displayed here.
- `LICENSE`: Textfile containing the license for this source code. You can find 
- `data/`: The directory `data` contains
    - `*.csv`: Different datasets provided by Gabriel Trierweiler Ribeiro for [[1]](#1),
    used for training, validation and test. Of particular interest are:
    - `CAT.csv`: Caterpillar stock price volatility.
    - `EBAY.csv`: Ebay stock price volatility.
    - `MSFT.csv`: Microsoft stock price volatility.
- `results/`
    - (Pre)-trained models and results as `sklearn.model_select.RandomizedSearchCV`
    objects.
    - For ARIMA, only the scalers are provided for now. The rest follows soon if 
    required.
    - ARIMA results still as CSV files.
- `.gitignore`: Command file for Github to ignore files with Python-specific extensions.

## Usage
The easiest way to reproduce the results is to run the Jupyter Notebooks. This is highly 
recommended, because this does not require a local installation.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/TUD-STKS/PyRCN-Benchmark/main)

To run the scripts or to start the Jupyter Notebook locally, at first, please ensure 
that you have a valid Python distribution installed on your system. Here, at least 
Python 3.8 is required.

Next, you need to clone the repository. please note that we require the remote 
repository PyESN to be cloned as well. To do so, please clone the repository using 
`clone --recurse-submodules https://github.com/TUD-STKS/PyRCN-Benchmark.git`. In that 
way, the directory `PyESN` does not remain empty.

You can then call `run_jupyter-lab.ps1` or `run_jupyter-lab.sh`. This will install a new 
[Python venv](https://docs.python.org/3/library/venv.html), which is our recommended way 
of getting started.

To manually reproduce the results, you should create a new Python venv as well.
Therefore, you can run the script `create_venv.sh` on a UNIX bash or `create_venv.ps1`
that will automatically install all packages from PyPI. Afterwards, just type 
`source .virtualenv/bin/activate` in a UNIX bash or `.virtualenv/Scripts/activate.ps1`
in a PowerShell.

The individual steps to reproduce the results should be in the same order as in the 
paper. Great would be self-explanatory names for each step.

At first, we define the datasets to be loaded and load them. They are already stored in 
`data`.

We restrict the data to the stock price volatility of the current day (`t0`).

```python
import pandas as pd
from collections import OrderedDict


    datasets = OrderedDict({"CAT": 0, "EBAY": 1, "MSFT": 2})
    data = [None] * len(datasets)

    for dataset, k in datasets.items():
        data[k] = pd.read_csv(f"./data/{dataset}.csv")["t0"].to_frame()
```


At first, we reproduce the HAR experiments. Essentially, this is a feature 
transformation of the time series by adding the moving average across 5 and 22 days to
the original dataset, which consequently has now three dimensions. We achieve that 
with the `sklearn.pipeline.FeatureUnion`. Afterwards, we normalize the inputs to be 
between 0 and 1, and regress from the HAR features to the target without regularization.

Since the target is normalized between 0 and 1 as well, we use the  

```python
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import TransformedTargetRegressor
from preprocessing import compute_average_volatility


day_volatility_transformer = FunctionTransformer(
                func=compute_average_volatility, kw_args={"window_length": 1})
week_volatility_transformer = FunctionTransformer(
    func=compute_average_volatility, kw_args={"window_length": 5})
month_volatility_transformer = FunctionTransformer(
    func=compute_average_volatility, kw_args={"window_length": 22})
har_features = FeatureUnion(
    transformer_list=[("day", day_volatility_transformer),
                      ("week", week_volatility_transformer),
                      ("month", month_volatility_transformer)])
har_pipeline = Pipeline(
    steps=[("har_features", har_features),
           ("scaler", MinMaxScaler()),
           ("lstsq", TransformedTargetRegressor(
               transformer=MinMaxScaler()))])
```

We optimize a model using a random search.

```python
from preprocessing import ts2super
import itertools
from model_selection import PredefinedTrainValidationTestSplit
from sklearn.model_selection import GridSearchCV


# Prepare data
df = pd.concat([ts2super(d, 0, H) for d in data])
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, -1].values.reshape(-1, 1)
test_fold = [
    [k] * len(ts2super(d, 0, H)) for k, d in enumerate(data)]
test_fold = list(itertools.chain.from_iterable(test_fold))

ps = PredefinedTrainValidationTestSplit(
    test_fold=test_fold, validation=False)

# Run model selection
search = GridSearchCV(
    estimator=har_pipeline, param_grid={}, cv=ps,
    scoring={"MSE": "neg_mean_squared_error",
             "RMSE": "neg_root_mean_squared_error", "R2": "r2"},
    refit="R2", return_train_score=True).fit(X, y)
```

We load and transform test data.

```python
from file_handling import load_data


test_data = load_data("../data/test.csv")
X = feature_trf.transform(test_data)
X_test = scaler.transform(X)
```

Finally, we predict the test data.

```python
y_pred = model.predict(X_test)
```

After you finished your experiments, please do not forget to deactivate the venv by 
typing `deactivate` in your command prompt.

The aforementioned steps are summarized in the script `main.py`. The easiest way to
reproduce the results is to either download and extract this Github repository in the
desired directory, open a Linux Shell and call `run.sh` or open a Windows PowerShell and
call `run.ps1`. 

In that way, again, a [Python venv](https://docs.python.org/3/library/venv.html) is 
created, where all required packages (specified by `requirements.txt`) are installed.
Afterwards, the script `main.py` is excecuted with all default arguments activated in
order to reproduce all results in the paper.

If you want to suppress any options, simply remove the particular option.

## Acknowledgements
This research was supported by Europäischer Sozialfonds (ESF), the Free State of Saxony
(Application number: 100327771) and Ghent University under the Special Research Award
number BOF19/PDO/134.

We kindly thank Gabriel Trierweiler Ribeiro for his support and expertise regarding the 
stock price volatility datasets.

## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

You can use the following BibTeX entry
```
@article{Steiner2022pyrcn,
	title = {PyRCN: A Toolbox for Exploration and Application of Reservoir Computing Networks},
	journal = {Engineering Applications of Artificial Intelligence},
	volume = {113},
	pages = {104964},
	year = {2022},
	issn = {0952-1976},
	doi = {10.1016/j.engappai.2022.104964},
	url = {https://www.sciencedirect.com/science/article/pii/S0952197622001713},
	author = {Peter Steiner and Azarakhsh Jalalvand and Simon Stone and Peter Birkholz},
}
```

## References
<a id="1">[1]</a> 
Gabriel Trierweiler Ribeiro, André Alves Portela Santos, Viviana Cocco Mariani, 
Leandro dos Santos Coelho. (2021). 
Novel hybrid model based on echo state neural network applied to the prediction of stock 
price return volatility. 
Expert Systems with Applications, 184, 115490. 
[10.1016/j.eswa.2021.115490](https://doi.org/10.1016/j.eswa.2021.115490)
