from Glaciers import Glacier, Model
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
import joypy
import pandas as pd
from rasterio.features import geometry_mask


def sample_indices(a, n, block=0, random_arr=None):

    if random_arr is None:
        random_arr = a

    values_indices = np.where(~np.isnan(random_arr))

    if block == 0:
        random_sample = np.random.choice(len(values_indices[0]), n)
        random_indices = [arr[random_sample] for arr in values_indices]

    elif block == 1:
        random_int = np.random.choice(len(values_indices[0]) - n, 1)
        random_sample = np.arange(random_int, random_int + n)
        random_indices = [arr[random_sample] for arr in values_indices]

    elif block == 2:



        if len(values_indices) == 3:
            values_indices = values_indices[1:]

        random_int = np.random.choice(len(values_indices[0]) - n, 1)
        random_sample = [arr[random_int] for arr in values_indices]

        y0 = int(random_sample[0])
        x0 = int(random_sample[1])

        w_u, w_l = 1, 1
        h_u, h_l = 1, 1

        while len(random_sample[0]) <= n:

            if x0 - w_l > 0:
                w_l += 1
            if x0 + w_u < a.shape[-1] - 2:
                w_u += 1
            if y0 - h_l > 0:
                h_l += 1
            if y0 + h_u < a.shape[-2] - 2:
                h_u += 1

            random_sample = np.where(~np.isnan(a[y0 - h_l: y0 + h_u, x0 - w_l: x0 + w_u]))

        random_indices = random_sample[-2][:n] + y0 - h_l, random_sample[-1][:n] + x0 - w_l

    return random_indices


def compare_alpha(glacier, model, N, n, block):
    alphas = np.zeros(shape=(N, 1))
    errors = np.zeros(shape=(N, 1))

    a = glacier.true_thickness_im[0]
    outline = None

    if block == 2:
        outline = ~geometry_mask(glacier.outline.geometry, glacier.true_thickness_im[0].shape,
                                 glacier.meta['transform'], all_touched=False) * 1.0
        outline[outline == 0] = np.NaN

    for i in range(N):
        random_indices = sample_indices(a, n, block=block, random_arr=outline)

        x = model.thickness[0][tuple(random_indices)]
        y = glacier.true_thickness_im[0][tuple(random_indices)]

        x, y = x[~np.isnan(x)], y[~np.isnan(x)]
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]

        alpha, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y[:, np.newaxis], rcond=None)

        mod_im = alpha * model.thickness
        error = mod_im - glacier.true_thickness_im

        alphas[i] = alpha
        errors[i] = np.nanmean(error)

    return alphas, errors


def alpha_joyplot(glacier, model, nbr_of_iterations, nbr_of_samples, n_step, block):
    nbr_of_plots = int(np.floor(len(glacier.true_thickness_array) / n_step))
    nbr_of_samples_array = np.zeros(shape=(nbr_of_plots, 1))

    x = model.thickness_array
    y = glacier.true_thickness_array

    x, y = x[~np.isnan(x)], y[~np.isnan(x)]
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]

    true_alpha, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y[:, np.newaxis], rcond=None)
    alphas = np.zeros(shape=(nbr_of_iterations, nbr_of_plots))
    errors = np.zeros_like(alphas)

    for i in range(nbr_of_plots):
        alphas[:, i: i + 1], errors[:, i: i + 1] = compare_alpha(glacier, model,
                                                                 nbr_of_iterations, nbr_of_samples, block=block)
        nbr_of_samples_array[i] = nbr_of_samples
        nbr_of_samples += n_step

    alphas = pd.DataFrame(alphas, columns=nbr_of_samples_array[:, 0].astype('int'))

    labels = [f'n = {i}, $\\bar{{\\alpha_{{GPR}}}}$ = {alphas[i].mean():2.2f}, ' \
              f'$\\sigma$ = {alphas[i].std():2.2f}'
              for i in nbr_of_samples_array[:, 0].astype('int')]

    alphas.columns = labels

    plt.figure()
    fig, axs = joypy.joyplot(alphas, ylim='own', overlap=0.5, title='\n\n\n\n',
                             figsize=(10, 5))
    fig.suptitle(f'Distribution of $\\alpha_{{GPR}}$ as a function of the number of sampled cells\n'
                 f'$\\alpha_{{GPR}}$ = {true_alpha[0][0]:2.2f}, {nbr_of_iterations} iterations, '
                 f'{len(x)} cells{", clumped" if block else ""}\n'
                 f'{glacier.name}')
    fig.savefig(f'alpha_joyplot_{glacier.name}_{block}.png')


def animate_samples(glacier, n, N, n_step, block=1, save_output=''):
    fig = plt.figure()
    ax = plt.gca()

    a = glacier.true_thickness_im[0]
    outline = None

    if block == 2:
        outline = ~geometry_mask(glacier.outline.geometry, glacier.true_thickness_im[0].shape,
                                 glacier.meta['transform'], all_touched=False) * 1.0
        outline[outline == 0] = np.NaN

    nbr_of_plots = int(np.floor(len(glacier.true_thickness_array) / n_step))
    index_array = [sample_indices(a, n + i * n_step, block, random_arr=outline) for i in range(nbr_of_plots) for j in range(N)]

    b = [0, 2, 1, 3]

    extent = [glacier.extent[i] for i in b]
    mask = np.zeros(shape=a.shape)
    mask[mask == 0] = np.NaN
    mask[tuple(index_array[0])] = 1
    im = ax.imshow(mask, extent=glacier.extent)
    contour = glacier.outline.plot(ax=ax, facecolor='None', edgecolor='black')

    def update(i):
        ax.clear()
        mask = np.zeros(shape=a.shape)
        mask[mask == 0] = np.NaN
        mask[tuple(index_array[i])] = 1
        im = ax.imshow(mask, extent=extent)
        contour = glacier.outline.plot(ax=ax, facecolor='None', edgecolor='black')
        ax.set_title(f'n = {len(index_array[i][0])}')

        return [contour]

    anim = animation.FuncAnimation(fig, update, frames=len(index_array), repeat=False)

    #plt.show()
    anim.save(f'{glacier.tag}_animation_{block}.gif', writer='imagemagick',
              fps=int(nbr_of_plots / 2))  # , extra_args=['-vcodec', 'libx26'])


if __name__ == '__main__':
    folder = 'alpha_compare'
    path = '../Glacier_test_case'

    try:
        os.makedirs(folder)
    except OSError:
        pass

    ng = Glacier('North Glacier', 'north_glacier',
                 dem_path=f'{path}/north_glacier_dem.tif',  # Path of the dem, works best with .tif
                 outline_path=f'{path}/north_glacier_utm.shp',  # Path of the outline
                 gpr_path=f'{path}/north_glacier_gpr.xyz',  # Path of the GPR data, can be txt or a DataFrame
                 whitespace=True, header=None, img_folder=folder)  # Some more parameters, see documentation

    ng.add_model(Model(model_name='North Glacier, consensus, Farinotti (2019)',
                       model_path=f'{path}/ng_consensus.tif',
                       tag='farinotti_consensus'))

    nbr_of_iterations = 500
    nbr_of_samples = 50
    n_step = 100

    animate_samples(ng, nbr_of_samples, 15, n_step, block=2)

    """ alpha_joyplot(ng, ng.models['farinotti_consensus'], nbr_of_iterations, nbr_of_samples, n_step, block=0)
    alpha_joyplot(ng, ng.models['farinotti_consensus'], nbr_of_iterations, nbr_of_samples, n_step, block=1)
    alpha_joyplot(ng, ng.models['farinotti_consensus'], nbr_of_iterations, nbr_of_samples, n_step, block=2)"""
