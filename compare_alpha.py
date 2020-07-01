from Glaciers import Glacier, Model
import numpy as np
import os
import matplotlib.pyplot as plt
import joypy
import pandas as pd


def sample_indices(a, n, block=False):
    values_indices = np.where(~np.isnan(a))

    if block:
        random_int = np.random.choice(len(values_indices[0]) - n, 1)
        random_sample = np.arange(random_int, random_int + n)

    else:
        random_sample = np.random.choice(len(values_indices[0]), n)

    random_indices = [values_indices[i][random_sample] for i in range(len(values_indices))]

    return random_indices


def compare_alpha(glacier, model, N, n, block):
    alphas = np.zeros(shape=(N, 1))
    errors = np.zeros(shape=(N, 1))

    for i in range(N):
        random_indices = sample_indices(glacier.true_thickness_im, n, block=block)

        x = model.thickness[tuple(random_indices)]
        y = glacier.true_thickness_im[tuple(random_indices)]

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

    indices = sample_indices(ng.true_thickness_im, 50, False)

    mask_array = np.zeros(shape=ng.true_thickness_im.shape, dtype='int')
    mask_array[tuple(indices)] = 1
    ng.plot_map(mask_array, cbar_unit='', tag='mask', title='region', outline=True)

    nbr_of_iterations = 500
    nbr_of_samples = 50
    n_step = 100

    alpha_joyplot(ng, ng.models['farinotti_consensus'], nbr_of_iterations, nbr_of_samples, n_step, block=False)
    alpha_joyplot(ng, ng.models['farinotti_consensus'], nbr_of_iterations, nbr_of_samples, n_step, block=True)
