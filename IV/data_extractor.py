#! /usr/bin/env python
import glob
import logging
import os
import traceback
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib2tikz import save as tikz_save
from scipy.constants import Boltzmann as k
from scipy.constants import elementary_charge as q
from scipy.optimize import curve_fit

from IV.devices import *

colors = []
for color in mpl.rcParams['axes.prop_cycle']:
    colors = colors + list(color.values())



def surface_recombination_plot(df, voltage, legend= False):
    fig, ax = plt.subplots()
    perimeter_area = []
    surf_rec = []
    for index, row in best_large_devices_surf_rec.iterrows():
        perimeter_area.append(row.perimeter/row.area)
        surf_rec.append(row.surf_rec[1])
    a, b = np.polyfit(perimeter_area, surf_rec, 1)
    x = np.array([0,1000])
    ax.plot(x,a*x + b,"--", linewidth=2,zorder = -1,label = "Fit = a*x + b :\n a = " '{:.1e}'.format(a) +" b="+ '{:.1e}'.format(b) )
    if legend:
        ax.legend()
    for index, row in best_large_devices_surf_rec.iterrows():
        ax.scatter(row.perimeter/row.area, row.surf_rec[1],label ="$"+ index+"$", alpha = 1, color = data_extractor.colors[1])
    # ax.set_yscale("log")
    ax.set_ylim(0,1)
    ax.set_xlim(0,1000)
    # ax.legend()
    ax.ticklabel_format(style="sci", axis="y", scilimits=(-2,2))
    ax.set_xlabel("Perimeter/Area ($cm^{-1}$)")
    ax.set_ylabel("Current Density ($cm^{-1}$)")
    return fig, ax, (a, b)

def J02_fit(df, parameters=True, legend= False):
    fig, ax = plt.subplots()
    J02s = []
    perimeter_over_areas = []
    for index, row in df.iterrows():
        x = np.linspace(0.4,1,num=61)
        popt, pcov = curve_fit(double_diode_model,x,row.current_density[140:201],p0=(1e-10,1e-12))
        curve_y = double_diode_model(x, popt[0],popt[1])
        J02s.append(popt[1])
        perimeter_over_areas.append(row.perimeter/row.area)
    ax.scatter(perimeter_over_areas, J02s)
    ax.set_ylim(1e-11,0.4e-8)
    J02_perimeter, J02_bulk = np.polyfit(perimeter_over_areas, J02s, 1)
    x_values= np.array([0,800])
    n_i = 2.1e6
    S0LS = J02_perimeter/(q*n_i)
    j02_bulk_string = "{0:.2e}".format(J02_bulk)
    j02_perimeter_string = "{0:.2e}".format(J02_perimeter)
    S0LS_string = "{0:.1f}".format(S0LS)
    label = "Bulk $J_{02}$ = "+ j02_bulk_string+" $A \ cm^{-2}$"+"\n"+"Perimeter $J_{02}$ = "+j02_perimeter_string +" $A \ cm^{-2}$" + "\n" + "$S_0L_s$ = " + S0LS_string + " $cm^2 \ s^{-1}$"
    # label = "J_{{02-bulk}} = \%.3E $A cm^{{-2}}$ - J_{{02-perimeter}} = \%.3E $A cm^{{-2}}$" %(J02_bulk, J02_perimeter)
    ax.plot(x_values, J02_perimeter*x_values + J02_bulk,"--", label = label, color="gray",zorder=-1,)
    if legend:
        ax.legend()
    n_i = 2.1e6
    S0LS = J02_perimeter/(q*n_i)
    if parameters:
        return fig, ax, J02_perimeter, J02_bulk, S0LS
    else:
        return fig, ax
    
def double_diode_fit_plot(df, parameters=False):
    fig, ax = plt.subplots()
    if isinstance(df,pd.DataFrame):
        for index, row in df.iterrows():
            x = np.linspace(0.4,1,num=61)
            popt, pcov = curve_fit(double_diode_model,x,row.current_density[140:201],p0=(1e-10,1e-12))
            curve_y = double_diode_model(x, popt[0],popt[1])
            ax.plot(df.voltage, df.current_density)
            j01_string = "{0:.2e}".format(popt[0])
            j02_string = "{0:.2e}".format(popt[1])
            label = "$J_{01} = $" + j01_string +  " $A \ cm^{-2}$" +"\n"+ "$J_{02} = $" +j02_string+ " $A \ cm^{-2}$"
            ax.plot(x, curve_y, '--', label = label)
            ax.legend()
            ax.set_xlabel("Voltage (V)")
            ax.set_ylabel("Current Density ($A \ cm^{-2}$)")
        if parameters:
            return fig, ax, J02_perimeter, J02_bulk, S0LS
        else:
            return fig, ax
    else:
        x = np.linspace(0.4,1,num=61)
        popt, pcov = curve_fit(double_diode_model,x,df.current_density[140:201],p0=(1e-10,1e-12))
        curve_y = double_diode_model(x, popt[0],popt[1])
        ax.plot(df.voltage[140:201], df.current_density[140:201])
        j01_string = "{0:.2e}".format(popt[0])
        j02_string = "{0:.2e}".format(popt[1])
        label = "$J_{01} = $" + j01_string +  " $A \ cm^{-2}$" +"\n"+ "$J_{02} = $" +j02_string+ " $A \ cm^{-2}$"
        ax.plot(x, curve_y, '--', label = label)
        ax.legend()
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current Density ($A \ cm^{-2}$)")
        if parameters:
            return fig, ax, J02_perimeter, J02_bulk, S0LS
        else:
            return fig, ax

def create_PV_dataframe(pkl_name, epi, filepath=os.getcwd(), delimeter=',', light=False,  dev_loc="on_chip", force_analysis=False):
    """Create dataframe of measurements in a filepath. If light measurements then light = True.
     Requires name of pickle to be defined and the epistructure"""
    data_folder = os.path.join(filepath, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print("Made new data Folder")
    if not os.path.isfile(os.path.join(data_folder, pkl_name)) or force_analysis == True:
        print("Pickle does not exist")
        list_of_measurements_dfs = []
        for file in DataExtractor.list_files_by_type(filepath):
            try:
                voltage, current_a = DataExtractor.data_split(
                    file, delimiter=delimeter)
                data = DataToDataFrame(voltage, current_a, file)
                data.epi, data.dev_loc = epi, dev_loc
                if light == False:
                    list_of_measurements_dfs.append(data.dark_data())
                else:
                    list_of_measurements_dfs.append(data.light_data())
            except Exception as e:
                logging.warning(traceback.print_exc())
                logging.warning(
                    "Error in dataframe with file {} in {} ".format(file, os.getcwd()))
                pass

        master_df = pd.concat(list_of_measurements_dfs)

        master_df.to_pickle(filepath + "\\data\\" + pkl_name)
        master_df.to_csv(filepath + "\\data\\" + pkl_name[:-4] + ".csv")
        return master_df
    else:
        print("Getting dataframe from data/pkl")
        return pd.read_pickle(os.path.join(data_folder, pkl_name))


def saveTikzPng(filename, watermark=None, thesis=False, show=False):
    if watermark is not None:
        plt.gcf().text(0.125, 0.9, watermark, fontsize=8)
    filename_png = filename + '.png'
    filename_pdf = filename + '.pdf'
    fig = plt.gcf()
    # plt.plot()
    d = os.getcwd()
    figure_folder = os.path.join(d, 'figures')
    tex_folder = os.path.join(d, 'tex')
    if not os.path.exists(tex_folder):
        os.makedirs(tex_folder)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    tikz_save(
        tex_folder + "/" + filename + '.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
    )
    if thesis == False:
        fig.savefig(figure_folder + "/" + filename_png,
                    format='png', dpi=600, bbox_inches='tight')
    else:
        fig.savefig(figure_folder + "/" + filename_pdf,
                    format='pdf', dpi=600, bbox_inches='tight')
        fig.savefig(figure_folder + "/" + filename_png,
                    format='png', dpi=600, bbox_inches='tight')         
    if show == True:
        plt.show()


def round_to_n(x, n):
    """Rounds number x to n digits"""
    return round(x, -int(np.floor(np.sign(x) * np.log10(abs(x)))) + n)


def remove_negative_values(a, b, column):
    """Removes negative values and corresponding values by column choice"""
    if column == 0 or column == 1:
        try:
            zipper = filter(lambda x: x[column] >= 0, zip(a, b))
            a, b = zip(*zipper)
            return np.array(a), np.array(b)
        except Exception as e:
            logging.warning(traceback.print_exc())
            logging.warning("Error in remove_negative_values")


def find_nearest(array, value):
    """Finds closest value to number in an array"""
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


class DataExtractor:

    """Get files from folder and get x and y data"""
    @staticmethod
    def data_split(filename, number_of_lines=0, delimiter=','):
        try:
            my_data = np.genfromtxt(
                filename, delimiter=delimiter, skip_header=number_of_lines)
            column1 = my_data[:, 0]
            column2 = my_data[:, 1]
            return column1, column2
        except Exception as e:
            logging.warning(traceback.print_exc())
            logging.warning("Error in data_split with file {} in {}".format(
                filename, os.getcwd()))

    @staticmethod
    def list_files_by_type(filepath, file_extension='dat'):
        os.chdir(filepath)
        string = "*." + file_extension
        files = [file for file in glob.glob(string)]
        return files


class DeviceInfo:
    def __init__(self, filename):
        self.filename = filename

    @property
    def incident_power_mw(self):
        try:
            if 'mw'  in self.filename.lower():
                before, after = self.filename.split('P')
                return float(after.split('m')[0])
                
            elif "ma" in self.filename.lower():
                before, after = self.filename.split('ma')
                current = int(before[-3:])
                current_to_power_dict ={110:8.6,130:21.8, 150:34.7,170:47.8,190:60.9,210:74,230:87.2,250:100.2}
                return current_to_power_dict.get(current, 0)
            else:
                return 0
                
            
        except ValueError:
            logging.warning(traceback.print_exc())
            logging.warning("Error in incident_power_mw with file {} in {} ".format(
                self.filename, os.getcwd()))
            pass

    @property
    def device_code(self):
        if "_" in self.filename:
            return self.filename.split('_')[0]
        else:
            return self.filename.split('.')[0]

    @property
    def meas_info(self):
        if "_" in self.filename:
            info_dat = self.filename.split('_')[1]
            info = info_dat.split('.')[0]
            return info
        else:
            return ''

    @property
    def device_properties(self):
        for x in range(len(devices)):
            device = devices[x]
            if self.device_code == device.name:

                return device
            else:
                continue


class DarkCalculations(DeviceInfo):
    def __init__(self, voltage, current_a, filename):
        DeviceInfo.__init__(self, filename)
        self.voltage = voltage
        self.current_a = current_a

    

    @property
    def current_density(self):
        return self.current_a / self.device_properties.area_cm

    def voltage_log_current(self, density=False):
        if density:
            voltage, current_density_a = remove_negative_values(
                self.voltage, self.current_density, 1)
            return voltage, np.log10(current_density_a)
        else:
            voltage, current_a = remove_negative_values(
                self.voltage, self.current_a, 1)
            return voltage, np.log10(current_a)

    def voltage_ln_current(self, density=False):
        if density:
            voltage_log, current_density_a = remove_negative_values(
                self.voltage, self.current_density, 1)
            return voltage_log, np.log(current_density_a)
        else:
            voltage_log, current_a = remove_negative_values(
                self.voltage, self.current_a, 1)
            return voltage_log, np.log(current_a)

    def r_series(self, start_of_fit=1.4, end_of_fit=1.6):
        start_of_fit = find_nearest(self.voltage, start_of_fit)[0]
        end_of_fit = find_nearest(self.voltage, end_of_fit)[0]
        voltage_fit = self.voltage[start_of_fit:end_of_fit]
        current_a_fit = self.current_a[start_of_fit:end_of_fit]
        fit = np.polyfit(voltage_fit, current_a_fit, 1)
        r_series_fit = np.multiply(fit[0], voltage_fit) + fit[1]
        r_series = (1 / fit[0])
        return r_series, voltage_fit, r_series_fit

    def I_0(self, start_of_fit=0.5, end_of_fit=0.9):
        voltage, log_current = self.voltage_log_current()
        start_of_fit = find_nearest(voltage, start_of_fit)[0]
        end_of_fit = find_nearest(voltage, end_of_fit)[0]
        voltage_fit = voltage[start_of_fit:end_of_fit]
        log_current_a_fit = log_current[start_of_fit:end_of_fit]
        fit = np.polyfit(voltage_fit, log_current_a_fit, 1)
        log_I0 = fit[1]
        I0 = 10**log_I0
        # I0 = round_to_n(I0, number_of_digits)
        log_current_a_fit = np.multiply(fit[0], voltage) + log_I0
        return log_I0, I0, voltage_fit, log_current_a_fit,

    @property
    def surf_rec_current(self):
        current_densities= []
        voltages = [0.6, 0.8, 1]
        for v in voltages:
            index_of_closest_value = find_nearest(self.voltage, v)[0]
            current_densities.append(
                 self.current_density[index_of_closest_value])
        return dict(zip(voltages,current_densities))

    def ideality_factor(self):
        T = 300
        voltage_ln, ln_current_a = self.voltage_ln_current()
        slope = np.diff(ln_current_a) / (voltage_ln[1] - voltage_ln[0])
        ideality_factor = (q / (k * T)) * (1 / slope)
        return np.append(ideality_factor, ideality_factor[-1])

    @property
    def J_0(self):
        return self.I_0()[1] / self.device_properties.area_cm


class LightCalculations(DarkCalculations):
    def __init__(self, voltage, current_a, filename):
        DarkCalculations.__init__(self, voltage, current_a, filename)

    @property
    def electrical_power_w(self):
        voltage, current_a = remove_negative_values(
            self.voltage, self.current_a, 0)
        return -voltage * current_a

    @property
    def max_power(self):
        max_index = np.argmax(self.electrical_power_w)
        max_value = np.max(self.electrical_power_w)
        return max_value, max_index

    @property
    def int_opt_power(self, photon_energy=1.534) :# 1.534
        
        return -self.current_a[1] * photon_energy

    @property
    def internal_efficiency(self):
        return max(self.electrical_power_w / self.int_opt_power)

    @property
    def efficiency(self):
        if self.incident_power_mw == 0:
            return float('nan')
        else:
            return 1000 * (self.max_power[0] / self.incident_power_mw)

    @property
    def Voc(self):
        index_of_closest_value_to_zero, closest_current_value_to_zero = find_nearest(
            np.array(self.current_a), 0)
        voltage_values_about_axis = self.voltage[index_of_closest_value_to_zero -
                                                 1:index_of_closest_value_to_zero + 1]
        current_values_about_axis = self.current_a[index_of_closest_value_to_zero -
                                                   1:index_of_closest_value_to_zero + 1]
        fit = np.polyfit(voltage_values_about_axis,
                         current_values_about_axis, 1)
        return -float(fit[1] / fit[0])

    @property
    def Isc(self):
        if 0 in self.voltage:
            return self.current_a[np.where(self.voltage == 0)][0]
        else:
            index_of_closest_value_to_zero, closest_current_value_to_zero = find_nearest(
                self.voltage, 0)
            return self.current_a[index_of_closest_value_to_zero]

    @property
    def Jsc(self):
        return self.Isc / self.device_properties.area_cm

    @property
    def Vmp(self):
        voltage, current_a = remove_negative_values(
            self.voltage, self.current_a, 0)
        return voltage[self.max_power[1]]

    @property
    def Imp(self):
        voltage, current_a = remove_negative_values(
            self.voltage, self.current_a, 0)
        return current_a[self.max_power[1]]

    @property
    def fill_factor(self):
        return self.Imp * self.Vmp / (self.Voc * self.Isc)


class DataToDataFrame(LightCalculations):
    dev_loc = ''
    epi = ''

    def __init__(self, voltage, current_a, filename):
        LightCalculations.__init__(self, voltage, current_a, filename)

    def dark_data(self):
        data = {
            'epi': self.epi,
            'r_series'				: self.r_series()[0],
            'j0'					: self.J_0,
            'surf_rec': [self.surf_rec_current],
            'diameter'				: self.device_properties.diameter_um,
            'voltage' 				: [self.voltage],
            'current' 				: [self.current_a],
            'current_density'		: [self.current_density],
            'voltage_log'			: [self.voltage_log_current()[0]],
            'log_current_density'	: [self.voltage_log_current(density=True)[1]],
            'ideality_factor'		: [self.ideality_factor()],
            'device_loc'			: self.dev_loc,
            'area'					: self.device_properties.area_cm,
            'perimeter'				: self.device_properties.perimeter_cm,
            'info': self.device_properties.info,
            'grid_coverage': self.device_properties.grid_coverage,
            'grid_type': self.device_properties.grid_type,
            'device_code': self.device_properties.name
        }
        index = self.filename[:-4]

        df = pd.DataFrame(data, index=[index])
        return df

    def light_data(self):
        data = {
            'epi': self.epi,
            'r_series'				: self.r_series()[0],
            'diameter'				: self.device_properties.diameter_um,
            'voltage' 				: [self.voltage],
            'current' 				: [self.current_a],
            'current_density'		: [self.current_density],
            'voltage_log'			: [self.voltage_log_current()[0]],
            'log_current_density'	: [self.voltage_log_current(density=True)[1]],
            'device_loc'			: self.dev_loc,
            'int_efficiency': self.internal_efficiency,
            'area'					: self.device_properties.area_cm,
            'int_opt_power': self.int_opt_power,
            'int_opt_power_dens': self.int_opt_power / self.device_properties.area_cm,
            'perimeter'				: self.device_properties.perimeter_cm,
            'info': self.device_properties.info,
            'grid_coverage': self.device_properties.grid_coverage,
            'grid_type': self.device_properties.grid_type,
            'device_code': self.device_properties.name,
            'incident_power'		: self.incident_power_mw,
            'incident_power_dens'	: self.incident_power_mw / (self.device_properties.area_cm * 1000),
            'Voc'					: self.Voc,
            'Isc'					: self.Isc,
            'efficiency'			: self.efficiency,
            'fill_factor'			: self.fill_factor,
            'Vmp'					: self.Vmp,
            'Imp'					: self.Imp,
            'Jsc'					: self.Jsc,
            'electrical_power'		: [self.electrical_power_w]

        }
        index = self.filename[:-4]

        df = pd.DataFrame(data, index=[index])
        return df
