"""
File handling utilities required to reproduce the results in the paper
'PyRCN: A Toolbox for Exploration and Application of Reservoir Computing
Networks'.
"""
# Authors: Peter Steiner <peter.steiner@tu-dresden.de>,
# License: BSD 3 clause

import pandas as pd


def export_results(results, filename):
    """Store the results as a csv file."""
    df = pd.DataFrame.from_dict(results)
    df.to_csv(filename, sep=',', index=False)
