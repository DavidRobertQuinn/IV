import matplotlib.pyplot as plt
from IV.data_extractor import colors, saveTikzPng


def plot_scatter(x, y, xlabel, ylabel, *dfs, log_x=None, log_y=None, fig_size=None, labels=None, xlims=None, ylims=None, save_name=None, watermark=None):
    """Plots scatter plots of variables x and y from optional number of data frames"""
    fig, ax = plt.subplots()
    if fig_size:
        fig.set_size_inches(fig_size)
    for df, label in zip(dfs, labels):
        plt.scatter(df[x], df[y], label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        ax.legend()
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    if save_name:
        saveTikzPng(save_name, watermark)


def plot_lines(x, y, xlabel, ylabel, *dfs, log_x=None, log_y=None, fig_size=None, labels=None, xlims=None, ylims=None, save_name=None, watermark=None):
    """Plots scatter plots of vraibles x and y from optional number of data frames"""
    fig, ax = plt.subplots()
    if fig_size:
        fig.set_size_inches(fig_size)
    isc_min_list = []
    jsc_min_list = []
    for count, df in enumerate(dfs):
        isc_min_list.append(min(df.Isc))
        jsc_min_list.append(min(df.Jsc))
        for index, row in df.iterrows():
            plt.plot(row[x], row[y], color=colors[count])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if labels:
        for count, label in enumerate(labels):
            plt.plot([], [], colors[count], label=label)
        ax.legend()
    ax.set_xlim(0, 1.4)
    ax.set_ylim(min(isc_min_list) * 1.1, -0.2 * min(isc_min_list))
    if y == 'current_density':
        ax.set_ylim(min(jsc_min_list) * 1.1, -0.2 * min(jsc_min_list))
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if save_name:
        saveTikzPng(save_name, watermark=watermark)
