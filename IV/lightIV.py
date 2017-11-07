#! /usr/bin/env python
import matplotlib.pyplot as plt
from IV.data_extractor import colors, saveTikzPng

def drop_dark_data(df):
    """Return dataframe in which the index does not contain the string 'dark' """
    s = df.index.str.contains("dark")
    return df[~s].copy()

def add_suns(df):
    """adds intensity in suns columns to dataframe"""
    df['suns'] = df.int_opt_power / (df.area * .1)

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

def full_analysis(*dfs, labels=None,  watermark=None, base_save_name=None,one_sun_x_lims= (100,1000) ):
    if labels == None:
        labels = [None]*len(dfs)
    updated_dfs = []
    for df in dfs:
        updated_df = drop_dark_data(df)
        add_suns(updated_df)
        updated_dfs.append(updated_df)
    if base_save_name:
        plot_lines('voltage', 'current', 'Voltage (V)', 'Current (A)', *updated_dfs,
                   labels=labels, save_name="IV" + base_save_name, watermark=watermark)
        plot_lines('voltage', 'current_density', 'Voltage (V)',
                   'Current Density  ($A \ cm^{-2}$)', *updated_dfs, labels=labels, save_name="JV" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'int_efficiency', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$\eta_{int}$",
                     *updated_dfs, labels=labels, save_name="eta_suns" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'Isc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$I_{sc}$ (A)",
                     *updated_dfs, labels=labels, save_name="eta_Isc" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'Jsc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$J_{sc}$ ($A \ cm^{-2}$)",
                     *updated_dfs, labels=labels, save_name="eta_Jsc" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'fill_factor', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "FF",
                     *updated_dfs, labels=labels, save_name="eta_FF" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'Voc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$V_{oc}$ (V)", *updated_dfs,
                     log_x=True, labels=labels, xlims=one_sun_x_lims, save_name="eta_voc" + base_save_name, watermark=watermark)
    else:
        plot_lines('voltage', 'current', 'Voltage (V)',
                   'Current (A)', *updated_dfs, labels=labels)
        plot_lines('voltage', 'current_density', 'Voltage (V)',
                   'Current Density  ($A \ cm^{-2}$)', *updated_dfs, labels=labels)
        plot_scatter('suns', 'int_efficiency',
                     "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$\eta_{int}$", *updated_dfs, labels=labels)
        plot_scatter(
            'suns', 'Isc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$I_{sc}$ (A)", *updated_dfs, labels=labels)
        plot_scatter(
            'suns', 'Jsc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$J_{sc}$ ($A \ cm^{-2}$)", *updated_dfs, labels=labels)
        plot_scatter('suns', 'fill_factor',
                     "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "FF", *updated_dfs, labels=labels)
        plot_scatter('suns', 'Voc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)",
                     "$V_{oc}$ (V)", *updated_dfs, xlims=one_sun_x_lims, log_x=True, labels=labels)
