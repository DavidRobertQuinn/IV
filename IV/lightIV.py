#! /usr/bin/env python
from IV.plot_data import plot_lines, plot_scatter


def drop_dark_data(df):
    """Return dataframe in which the index does not contain the string 'dark' """
    s = df.index.str.contains("dark")
    return df[~s].copy()


def add_suns(df):
    """adds intensity in suns columns to dataframe"""
    df['suns'] = df.int_opt_power / (df.area * .1)


def full_analysis(*dfs, labels=None,  watermark=None, base_save_name=None, one_sun_x_lims=(100, 1000)):
    if labels == None:
        labels = [None] * len(dfs)
    updated_dfs = []
    for df in dfs:
        updated_df = drop_dark_data(df)
        add_suns(updated_df)
        updated_dfs.append(updated_df)
    if base_save_name:
        plot_lines('voltage', 'current', 'Voltage (V)', 'Current (A)', *updated_dfs,
                   labels=labels, save_name=base_save_name + "_v_i",  watermark=watermark)
        plot_lines('voltage', 'current_density', 'Voltage (V)',
                   'Current Density  ($A \ cm^{-2}$)', *updated_dfs, labels=labels, save_name="JV" + base_save_name, watermark=watermark)
        plot_scatter('suns', 'int_efficiency', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$\eta_{int}$",
                     *updated_dfs, labels=labels, save_name=base_save_name + "_suns_int_eta", watermark=watermark)
        plot_scatter('suns', 'Isc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$I_{sc}$ (A)",
                     *updated_dfs, labels=labels, save_name=base_save_name + "_suns_Isc", watermark=watermark)
        plot_scatter('suns', 'Jsc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$J_{sc}$ ($A \ cm^{-2}$)",
                     *updated_dfs, labels=labels, save_name=base_save_name + "_suns_Jsc", watermark=watermark)
        plot_scatter('suns', 'fill_factor', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "FF",
                     *updated_dfs, labels=labels, save_name=base_save_name + "_suns_FF", watermark=watermark)
        plot_scatter('suns', 'Voc', "Power Density in Suns($0.1 \ W \ cm^{-2} $)", "$V_{oc}$ (V)", *updated_dfs,
                     log_x=True, labels=labels, xlims=one_sun_x_lims, save_name=base_save_name + "_suns_voc", watermark=watermark)
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
