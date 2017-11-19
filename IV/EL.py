import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
from mpl_toolkits.mplot3d import Axes3D
from skimage.color import rgb2gray


def oneD_intensity_profile(image_file, x0, y0, x1, y1, num_of_coords_on_line=500, image_ticks=False,
                           plot_ylabel=None, plot_ticks=True, save=False, show=False, figsize=None):
    image = scipy.misc.imread(image_file)
    gray_img = rgb2gray(image)
    x, y = np.linspace(x0, x1, num_of_coords_on_line), np.linspace(
        y0, y1, num_of_coords_on_line)
    # Extract the values along the line, using cubic interpolation
    zi = scipy.ndimage.map_coordinates(gray_img, np.vstack((y, x)))
    fig, axes = plt.subplots(nrows=2, figsize=figsize)
    axes[0].imshow(gray_img)
    axes[0].plot([x0, x1], [y0, y1], 'ro-')
    axes[0].axis('image')
    axes[1].plot(zi)
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
        plt.savefig(str(image_file[:-4] + '1D_intensity_profile' +
                        '.pdf'), format='pdf', dpi=600, bbox_inches='tight')
    else:
        return fig
