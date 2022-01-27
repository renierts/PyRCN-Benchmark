"""
Visualization utilities required to reproduce the results in the paper
'PyRCN: A Toolbox for Exploration and Application of Reservoir Computing
Networks'.
"""
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn.model_selection._search import BaseSearchCV
from typing import Union, List, Dict


def visualize_fit_and_score_time(search: BaseSearchCV, ax: Axes,
                                 id_vars: Union[str, List[str]],
                                 value_vars: Union[str, List[str]],
                                 **kwargs: Dict) -> Axes:
    df = pd.DataFrame(search.cv_results_)
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                        var_name='cols', value_name='vals')
    sns.pointplot(data=df_melted, x=id_vars,
                  y="vals", hue="cols", ax=ax, **kwargs)
    ax.set(xlabel="Reservoir size", ylabel="Time in seconds")
    ax.legend().set_title('')
    return ax
