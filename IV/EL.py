import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray
import os


def twoD_coloured_visualisation(image_file):
    parent_folder = os.path.dirname(image_file)
    image = scipy.misc.imread(image_file)
    gray_img = rgb2gray(image)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(gray_img, extent=(0, 1, 1, 0))
    ax.axis('tight')
    ax.axis('off')
    figure_folder = os.path.join(parent_folder, 'figures')
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    file_name = os.path.basename(
        image_file)[:-4] + '_2D_visualisation' + '.png'
    save_name = os.path.join(figure_folder, file_name)
    plt.savefig(save_name, format='png', dpi=600,
                bbox_inches='tight', pad_inches=0)


def oneD_intensity_profile(image_file, x0, y0, x1, y1, num_of_coords_on_line=500, image_ticks=False,
                           plot_ylabel=None, plot_ticks=False, save=False, show=False, figsize=None):
    image = scipy.misc.imread(image_file)
    gray_img = rgb2gray(image)
    x, y = np.linspace(x0, x1, num_of_coords_on_line), np.linspace(
        y0, y1, num_of_coords_on_line)
    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(gray_img, np.vstack((y, x)))
    fig, axes = plt.subplots(nrows=2, figsize=figsize)
    axes[0].imshow(gray_img, )
    axes[0].plot([x0, x1], [y0, y1], 'ro-', linewidth=2, markersize=5)
    axes[0].axis('image')
    axes[1].plot(zi, linewidth=2)
    axes[1].set_xticks([])
    axes[1].set_ylabel("Intensity (a.u) \n along red line")
    if not image_ticks:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    if not plot_ticks:
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    if plot_ylabel:
        axes[1].set_ylabel(plot_ylabel)
    if show:
        plt.show()
    if save:
        fig.savefig(str(image_file[:-4] + '_1D_intensity_profile' +
                        '.pdf'), format='pdf', dpi=600, bbox_inches='tight')
        fig.savefig(str(image_file[:-4] + '_1D_intensity_profile' +
                        '.png'), format='png', dpi=600, bbox_inches='tight')
    else:
        return fig,axes


def el_visualisation(image_file):
    image = scipy.misc.imread(image_file)
    gray_img = rgb2gray(image)
    gray_img = scipy.misc.imresize(gray_img, 0.3, interp='cubic')
    xx, yy = np.mgrid[0:gray_img.shape[0], 0:gray_img.shape[1]]
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection='3d')
    label = image_file[:-4] + "3D_Visualisation"
    ax.plot_surface(xx, yy, gray_img / np.max(gray_img),
                    rstride=1, cstride=1, cmap='viridis', linewidth=0)
    fake2Dline = mpl.lines.Line2D(
        [0], [0], linestyle="none", c='b', marker='o')
    plt.axis('off')
    # ax.legend([fake2Dline], [label], numpoints = 1)
    ax.view_init(elev=90, azim=-90)
    fig.savefig(label + '.png', format='png', dpi=600, bbox_inches='tight')
    fig.savefig(label + '.pdf', format='pdf', dpi=600, bbox_inches='tight')
    ax.view_init(elev=75, azim=-45)
    fig.savefig(str(label + 'angle' + '.png'),
                format='png', dpi=600, bbox_inches='tight')
    fig.savefig(str(label + 'angle' + '.pdf),
                format='pdf', dpi=600, bbox_inches='tight')
