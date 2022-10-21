#  Copyright (c) 2022 Szymon Mikler

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from benchmarks.tagged_collection import TaggedCollection

sns.set_style("darkgrid")


def plot(
    df: pd.DataFrame,
    x_axis,
    y_axis,
    compare,
    reference_y=None,
    title="UNKNOWN",
    xlabel=None,
    ylabel=None,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    linewidth: int | None = None,
):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True, dpi=300)

    if reference_y:
        ys = df.apply(lambda row: reference_y[row[X]], axis=1)
        print(ys)
        df[Y] = (df[Y] - ys) / ys

    def t(x):
        return np.percentile(x, q=20), np.percentile(x, q=80)

    sns.lineplot(
        ax=ax,
        data=df,
        x=x_axis,
        y=y_axis,
        estimator="median",
        err_kws={"alpha": 0.15},
        hue=compare,
        # hue=None,
        # errorbar=t,
        errorbar=("pi", 50),
        # errorbar=lambda x: 1,
        # errorbar=None,
        # markers=True,
    )

    if linewidth is not None:
        for line in ax.lines:
            line.set_linewidth(2)

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)


#########################

X = "data.layers"
Y = "data.throughput"

filters = (lambda row: "call_optimization" in row.tags,)

tagged = TaggedCollection.from_dllogs("benchmarks/good_logs/bench1.jsonl")

tagged = tagged.filter(*filters)
assert len(tagged) > 0, "Empty query!"

references = tagged["vanilla"]["call_optimization"]
reference_y_for_x = {}

for x, reference_for_x in references.groupby(lambda row: row[X], return_keys=True):
    avg = np.median(reference_for_x.get(Y))
    reference_y_for_x[x] = avg

    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), constrained_layout=True, dpi=300)

names = {
    "oct20,vanilla,call_optimization": "Inheriting from nn.Module",
    "oct20,sequential,call_optimization": "nn.Sequential",
    "oct20,symbolic,call_optimization": "SymbolicModel",
}

for tags, group in tagged.groupby("tags", return_keys=True):
    assert tags in names
    name = names[tags]
    for row in group:
        row["Definition"] = name

plot(
    tagged.df,
    x_axis=X,
    y_axis=Y,
    compare="Definition",
    reference_y=None,
    title="Performance comparison between different definitions of deep linear models",
    xlabel="Number of linear layers",
    ylabel="Batches per second (more is better)",
    ax=ax[0],
    fig=fig,
)
plot(
    tagged.df,
    x_axis=X,
    y_axis=Y,
    compare="Definition",
    reference_y=reference_y_for_x,
    title="Close-up",
    xlabel="Number of linear layers",
    ylabel="Throughput difference (%)",
    ax=ax[1],
    fig=fig,
)
plt.savefig("latest.png")
plt.show()
