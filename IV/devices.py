from collections import namedtuple
from math import pi
global devices

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
C7A2 = Device(name='C7A2', area_cm=pi * 0.015 ** 2, diameter_um=300, perimeter_cm=pi*.03, grid_coverage=0.011777,
            grid_type='radial+spokes', shape="circle", info='no_bond_metal_over_edge')

C7A1I = Device(name='C7A1I', area_cm=pi * 0.015 ** 2, diameter_um=300, perimeter_cm=pi * .03, grid_coverage=0.011777,
                grid_type='radial+spokes', shape="circle", info='ITO')

C7A3 = Device(name='C7A3', area_cm=pi * 0.015 ** 2, diameter_um=300, perimeter_cm=pi * .03, grid_coverage=0.011777,
                grid_type='inverted_square_thick_grid_n_contact', shape="circle", info='ITO')

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
devices = [
    C1A, C1A1, C1B,
    S1A1, S1A2, S1A3, S1A1F, S1A4,
    C2A, C2B,
    C3A1, C3A2,
    S3A1, S3A2, S3A3, S3A21,
    C4A1, C4A2,
    C6A1, C6A1F, C7A1, C7A2,C7A3,
    A4, A3, A2, A1, A1S, A0,
    B3, B2, B2F, B1S, B1, B0
        ]
