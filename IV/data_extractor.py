#! /usr/bin/env python
import os
import glob
import numpy as np
import traceback
import re
from collections import namedtuple
from matplotlib2tikz import save as tikz_save
from scipy.constants import elementary_charge, Boltzmann as q, k
from math import pi
import matplotlib.pyplot as plt
import logging
import pandas as pd
import sys
import matplotlib as mpl
desired_width = 320
pd.set_option('display.width', desired_width)
from scipy.optimize import curve_fit
import time
from tkinter import *
from tkinter.filedialog import askdirectory




colors = []
for color in mpl.rcParams['axes.prop_cycle']:
    colors = colors + list(color.values())

def create_PV_dataframe(pkl_name,epi,filepath='DEFAULT',delimeter = ',',light=False,  dev_loc= "on_chip"):
    """Create dataframe of measurements in a filepath. If light measurements then light = True.
     Requires name of pickle to be defined and the epistructure"""
    here =os.getcwd()
    data_folder = os.path.join(here, 'data')
    if not os.path.isfile(os.path.join(data_folder, pkl_name)):
        if filepath =='DEFAULT':
            root = Tk()
            root.update()
            filepath = askdirectory()
            root.destroy()
        list_of_measurements_dfs=[]
        for file in DataExtractor.list_files_by_type(filepath):

            try:
                voltage, current_a = DataExtractor.data_split(file, delimiter=delimeter)
                data = DataToDataFrame(voltage,current_a,file)
                data.epi, data.dev_loc = epi, dev_loc
                if light == False:
                    list_of_measurements_dfs.append(data.dark_data())
                else:
                    list_of_measurements_dfs.append(data.light_data())
            except Exception as e:
                logging.warning(traceback.print_exc())
                logging.warning("Error in dataframe with file {} in {} ".format(file, os.getcwd()))
                pass
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        master_df = pd.concat(list_of_measurements_dfs)
        
        master_df.to_pickle(filepath+r"/data/"+pkl_name)
        master_df.to_csv(filepath+r"/data/"+pkl_name[:-4] + ".csv" )
        return master_df
    else:
        return pd.read_pickle(os.path.join(data_folder, pkl_name))

def saveTikzPng(filename, watermark=None, thesis = False, show=False):
    if watermark is not None:
        plt.gcf().text(0.125, 0.9, watermark, fontsize=8)
    filename_png = filename + '.png'
    filename_pdf = filename + '.pdf'
    plt.gcf()
    plt.plot()
    d = os.getcwd()
    figure_folder = os.path.join(d, 'figures')
    tex_folder = os.path.join(d, 'tex')
    if not os.path.exists(tex_folder):
        os.makedirs(tex_folder)
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    tikz_save(
        tex_folder +"/"+filename + '.tex',
        figureheight='\\figureheight',
        figurewidth='\\figurewidth'
    )
    if thesis == False:
        plt.savefig(figure_folder+"/"+filename_png, format='png', dpi=600, bbox_inches='tight')
    else:
        plt.savefig(figure_folder+"/"+filename_pdf, format='pdf', dpi=600, bbox_inches='tight')
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
    def data_split(filename, number_of_lines=0, delimiter = ','):
        try:
            my_data = np.genfromtxt(filename, delimiter=delimiter, skip_header=number_of_lines)
            column1 = my_data[:, 0]
            column2 = my_data[:, 1]
            return column1, column2
        except Exception as e:
            logging.warning(traceback.print_exc())
            logging.warning("Error in data_split with file {} in {}".format(filename, os.getcwd()))

    @staticmethod
    def list_files_by_type(filepath, file_extension = 'dat'):
        original_filepath = os.getcwd()
        os.chdir(filepath)
        string = "*."+file_extension
        files = [file for file in glob.glob(string) ]
        
        os.chdir(original_filepath)
        return files



class DeviceInfo:
    def __init__(self, filename):
        self.filename = filename

    @property
    def incident_power_mw(self):
        try:
            if 'mw' not in self.filename.lower():
                return 0
            else:
                before, after = self.filename.lower.split('P')
                return float(after.split('m')[0])
        except ValueError:
            logging.warning(traceback.print_exc())
            logging.warning("Error in incident_power_mw with file {} in {} ".format(self.filename, os.getcwd()))
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

    Device = namedtuple('Device', 'name area_cm diameter_um perimeter_cm grid_coverage grid_type shape info')
    ### New Mask
    C1A = Device(name='C1A', area_cm=pi*0.01**2, diameter_um=200, perimeter_cm=pi*.02, grid_coverage=0.049,
                grid_type='spokes', shape="circle", info='narrow_bottom_contact')
    C1A1 = Device(name='C1A1', area_cm=pi * 0.01 ** 2, diameter_um=200, perimeter_cm=pi*.02, grid_coverage=0.049,
                grid_type='inverted_square', shape="circle", info='')
    C1B = Device(name='C1B', area_cm=pi * 0.01 ** 2, diameter_um=200, perimeter_cm=pi * .02, grid_coverage=0.049,
                 grid_type='spokes', shape="circle", info='wide_n_contact')

    S1A1 = Device(name='S1A1', area_cm=0.02 ** 2, diameter_um=200, perimeter_cm=4*.02, grid_coverage=0.0454,
                grid_type='inverted_square', shape="square", info='inverted_square_grid_50um')
    S1A1F = Device(name='S1A1F', area_cm=0.02 ** 2, diameter_um=200, perimeter_cm=4 * .02, grid_coverage=1,
                  grid_type='inverted_square', shape="square", info='full_grid_coverage')
    S1A2 = Device(name='S1A2', area_cm=0.02 ** 2, diameter_um=200, perimeter_cm=4 * .02, grid_coverage=0.0594,
                  grid_type='inverted_square', shape="square", info='inverted_square_grid_37um')
    S1A3 = Device(name='S1A3', area_cm=0.02 ** 2, diameter_um=200, perimeter_cm=4 * .02, grid_coverage=0.985,
                  grid_type='inverted_square', shape="square", info='20um_opening_with_large_mirror')
    S1A4 = Device(name='S1A4', area_cm=0.02 ** 2, diameter_um=200, perimeter_cm=4 * .02, grid_coverage=0.985,
                  grid_type='inverted_square', shape="square", info='20um_opening_with_large_mirror')

    C2A = Device(name='C2A', area_cm=pi * 0.0075 ** 2, diameter_um=150, perimeter_cm=pi * .015, grid_coverage=0.08,
                 grid_type='spokes', shape="circle", info='')

    C2B = Device(name='C2B', area_cm=pi * 0.0075 ** 2, diameter_um=150, perimeter_cm=pi * .015, grid_coverage=0.08,
                 grid_type='spokes', shape="circle", info='wider_bottom_contact')

    C3A1 = Device(name='C3A1', area_cm=pi * 0.005 ** 2, diameter_um=100, perimeter_cm=pi * .01, grid_coverage=0.08,
                 grid_type='spokes', shape="circle", info='')

    C3A2 = Device(name='C3A2', area_cm=pi * 0.005 ** 2, diameter_um=100, perimeter_cm=pi * .01, grid_coverage=0.08,
                  grid_type='spokes', shape="circle", info='')

    S3A1 = Device(name='S3A1', area_cm=0.01 ** 2, diameter_um=100, perimeter_cm=0.01 * 4, grid_coverage=0.045,
                  grid_type='spokes', shape="square", info='50um_grid_lines')

    S3A2 = Device(name='S3A2', area_cm=0.01 ** 2, diameter_um=100, perimeter_cm=0.01 * 4, grid_coverage=0.035172,
                  grid_type='inverse_square', shape="square", info='50um_grid_lines')

    S3A21 = Device(name='S3A21', area_cm=0.01 ** 2, diameter_um=100, perimeter_cm=0.01 * 4, grid_coverage=0.035172,
                  grid_type='inverse_square', shape="square", info='45_degree_orientation')

    S3A3 = Device(name='S3A3', area_cm=0.01 ** 2, diameter_um=100, perimeter_cm=0.01 * 4, grid_coverage=0.8,
                  grid_type='inverse_square', shape="square", info='37um_grid_lines')

    C4A1 = Device(name='C4A1', area_cm=pi * 0.0025 ** 2, diameter_um=50, perimeter_cm=pi * .005, grid_coverage=0.08,
                  grid_type='no_grid', shape="circle", info='')

    C4A2 = Device(name='C4A2', area_cm=pi * 0.0025 ** 2, diameter_um=50, perimeter_cm=pi * .005, grid_coverage=0.08,
                  grid_type='no_grid', shape="circle", info='')

    C6A1 = Device(name='C6A1', area_cm=pi * 0.02 ** 2, diameter_um=400, perimeter_cm=pi*.04, grid_coverage=0.0126,
                grid_type='inverted_square_50um', shape="circle", info='bond_metal_over_edge')
    C6A1F = Device(name='C6A1F', area_cm=pi * 0.02 ** 2, diameter_um=400, perimeter_cm=pi*.04, grid_coverage=1,
                grid_type='full', shape="circle", info='bond_metal_over_edge')
    C7A1 = Device(name='C7A1', area_cm=pi * 0.015 ** 2, diameter_um=300, perimeter_cm=pi*.03, grid_coverage=0.011777,
                grid_type='radial+spokes', shape="circle", info='bond_metal_over_edge')

    C7A1I = Device(name='C7A1I', area_cm=pi * 0.015 ** 2, diameter_um=300, perimeter_cm=pi * .03, grid_coverage=0.011777,
                  grid_type='radial+spokes', shape="circle", info='ITO')

    # S3A3 = Device(name='S3A3', area_cm=0.01**2, diameter_um=100, perimeter_cm=0.01*4, grid_coverage=0.011777,
    #               grid_type='inverse_square', shape="square", info='37um_grid_lines')
    ## Old Mask


    A4 = Device(name='A4', area_cm=pi*0.02**2, diameter_um=400, perimeter_cm=pi*.04, grid_coverage=0.166,
                grid_type='spokes', shape="circle", info='')
    A3 = Device(name='A3', area_cm=pi*0.015**2, diameter_um=300, perimeter_cm=pi*.03, grid_coverage=0.151,
                grid_type='spokes', shape="circle", info='')
    A2 = Device(name='A2', area_cm=pi*0.01**2, diameter_um=200, perimeter_cm=pi*.02, grid_coverage=0.112,
                grid_type='none', shape="circle", info='no_grid')
    A1S = Device(name='A1S', area_cm=pi*0.005**2, diameter_um=100, perimeter_cm=pi*.01, grid_coverage=0.1735,
                grid_type='spokes', shape="circle", info='')

    A1 = Device(name='A1', area_cm=pi * 0.005 ** 2, diameter_um=100, perimeter_cm=pi * .01, grid_coverage=0.152,
                 grid_type='spokes', shape="circle", info='')
    A0 = Device(name='A0', area_cm=pi * 0.0025 ** 2, diameter_um=50, perimeter_cm=pi * .005, grid_coverage=0.262,
                grid_type='spokes', shape="circle", info='')

    B3 = Device(name='B3', area_cm=pi*0.015**2, diameter_um=300, perimeter_cm=pi*.03, grid_coverage=0.152,
                grid_type='spokes', shape="circle", info='')
    B2 = Device(name='B2', area_cm=pi*0.01**2, diameter_um=200, perimeter_cm=pi*.02, grid_coverage=0.112,
                grid_type='none', shape="circle", info='no_grid')
    B2F = Device(name='B2F', area_cm=pi * 0.01 ** 2, diameter_um=200, perimeter_cm=pi * .02, grid_coverage=1,
                grid_type='spokes', shape="circle", info='full_grid')


    B1S = Device(name='B1S', area_cm=pi * 0.005 ** 2, diameter_um=100, perimeter_cm=pi * .01, grid_coverage=0.197,
                 grid_type='spokes', shape="circle", info='')

    B1= Device(name='B1', area_cm=pi * 0.005 ** 2, diameter_um=100, perimeter_cm=pi * .01, grid_coverage=0.152,
                 grid_type='none', shape="circle", info='')

    B0 = Device(name='B0', area_cm=pi * 0.0025 ** 2, diameter_um=50, perimeter_cm=pi * .005, grid_coverage=0.328,
                grid_type='none', shape="circle", info='')

    global devices
    devices = [

               C1A, C1A1, C1B,
               S1A1, S1A2, S1A3, S1A1F, S1A4,
               C2A, C2B,
               C3A1, C3A2,
               S3A1, S3A2, S3A3, S3A21,
               C4A1, C4A2,
               C6A1, C6A1F, C7A1, C7A1,
               A4, A3, A2, A1, A1S, A0,
               B3, B2, B2F, B1S, B1, B0
               ]



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
        return self.current_a/self.device_properties.area_cm

    def voltage_log_current(self, density = False):
        if density:
            voltage, current_density_a = remove_negative_values(self.voltage, self.current_density, 1)
            return voltage, np.log10(current_density_a)
        else:
            voltage, current_a = remove_negative_values(self.voltage, self.current_a, 1)
            return voltage, np.log10(current_a)

    def voltage_ln_current(self, density=False):
        if density:
            voltage_log, current_density_a = remove_negative_values(self.voltage, self.current_density, 1)
            return voltage_log, np.log(current_density_a)
        else:
            voltage_log, current_a = remove_negative_values(self.voltage, self.current_density, 1)
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
        micro_current_values = []
        voltages = [0.6, 0.8, 1]
        for v in voltages:
            index_of_closest_value = find_nearest(self.voltage, v)[0]
            micro_current_values.append(1000000*self.current_a[index_of_closest_value])
        return dict(zip(voltages, micro_current_values))




    def ideality_factor(self):
        T = 300
        voltage_ln, ln_current_a = self.voltage_ln_current()
        slope = np.diff(ln_current_a) / (voltage_ln[1] - voltage_ln[0])
        ideality_factor = (q/(k*T))*(1/slope)
        return np.append(ideality_factor, ideality_factor[-1])

    @property
    def J_0(self):
        return self.I_0()[1]/self.device_properties.area_cm


class LightCalculations(DarkCalculations):
    def __init__(self, voltage, current_a, filename):
        DarkCalculations.__init__(self, voltage, current_a, filename)

    @property
    def electrical_power_w(self):
        voltage, current_a = remove_negative_values(self.voltage, self.current_a, 0)
        return -voltage*current_a

    @property
    def max_power(self):
        max_index = np.argmax(self.electrical_power_w)
        max_value = np.max(self.electrical_power_w)
        return max_value, max_index

    @property
    def int_opt_power(self, photon_energy=1.534):
        return -self.current_a[301] * photon_energy

    @property
    def internal_efficiency(self):
        return max(self.electrical_power_w/self.int_opt_power)

    @property
    def efficiency(self):
        if self.incident_power_mw == 0:
            return float('nan')
        else:
            return 1000*(self.max_power[0]/self.incident_power_mw)

    # @property
    # def Voc(self):
    #     index_of_closest_value_to_zero, closest_current_value_to_zero = find_nearest(self.current_a, 0)
    #     return self.voltage[index_of_closest_value_to_zero]

    @property
    def Voc(self):
        index_of_closest_value_to_zero, closest_current_value_to_zero = find_nearest(np.array(self.current_a), 0)
        voltage_values_about_axis = self.voltage[index_of_closest_value_to_zero-1:index_of_closest_value_to_zero+1]
        current_values_about_axis = self.current_a[index_of_closest_value_to_zero-1:index_of_closest_value_to_zero+1]
        fit = np.polyfit(voltage_values_about_axis, current_values_about_axis,1)
        return -float(fit[1]/fit[0])

    @property
    def Isc(self):
        if 0 in self.voltage:
            return self.current_a[np.where(self.voltage == 0)][0]
        else:
            index_of_closest_value_to_zero, closest_current_value_to_zero = find_nearest(self.voltage, 0)
            return self.current_a[index_of_closest_value_to_zero]

    @property
    def Jsc(self):
        return self.Isc / self.device_properties.area_cm

    @property
    def Vmp(self):
        voltage, current_a = remove_negative_values(self.voltage, self.current_a, 0)
        return voltage[self.max_power[1]]

    @property
    def Imp(self):
         voltage, current_a = remove_negative_values(self.voltage, self.current_a, 0)
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
            'epi'			        : self.epi,
            'r_series'				: self.r_series()[0],
            'j0'					: self.J_0,
            'surf_rec'              : [self.surf_rec_current],
            'diameter'				: self.device_properties.diameter_um,
            'voltage' 				: [self.voltage],
            'current' 				: [self.current_a],
            'current_density'		: [self.current_density],
            'voltage_log'			: [self.voltage_log_current()[0]],
            'log_current_density'	: [self.voltage_log_current(density = True)[1]],
            'ideality_factor'		: [self.ideality_factor()],
            'device_loc'			: self.dev_loc,
            'area'					: self.device_properties.area_cm ,
            'perimeter'				: self.device_properties.perimeter_cm,
            'info'                  : self.device_properties.info,
            'grid_coverage'         : self.device_properties.grid_coverage,
            'grid_type'             : self.device_properties.grid_type,
            'device_code'           : self.device_properties.name
        }
        index = self.filename[:-4]

        df = pd.DataFrame(data, index=[index])
        return df


    def light_data(self):
        data = {
            'epi'			        : self.epi,
            'r_series'				: self.r_series()[0],
            'diameter'				: self.device_properties.diameter_um,
            'voltage' 				: [self.voltage],
            'current' 				: [self.current_a],
            'current_density'		: [self.current_density],
            'voltage_log'			: [self.voltage_log_current()[0]],
            'log_current_density'	: [self.voltage_log_current(density =True)[1]],
            'device_loc'			: self.dev_loc,
            'int_efficiency'	    : self.internal_efficiency,
            'area'					: self.device_properties.area_cm  ,
            'int_opt_power'         : self.int_opt_power,
            'int_opt_power_dens'    : self.int_opt_power/self.device_properties.area_cm ,
            'perimeter'				: self.device_properties.perimeter_cm,
            'info'                  : self.device_properties.info,
            'grid_coverage'         : self.device_properties.grid_coverage,
            'grid_type'             : self.device_properties.grid_type,
            'device_code'           : self.device_properties.name,
            'incident_power'		: self.incident_power_mw,
            'incident_power_dens'	: self.incident_power_mw/(self.device_properties.area_cm*1000),
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

