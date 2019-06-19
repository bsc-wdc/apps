import sys
import os
import numpy as np
import warnings
import numba
import datetime
from numba import vectorize
from numba import cuda
from numba import int32, int64, float32, float64
from numba.cuda.cudadrv import driver

from pycompss.api.task import task
from pycompss.api.parameter import *
from pycompss.api.constraint import constraint
from pycompss.api.implement import implement
from pycompss.api.api import compss_wait_on
from pycompss.api.api import compss_barrier


def main():
    dataset_folder = sys.argv[1]
    mesh_file = sys.argv[2]
    regions_file = sys.argv[3]
    output_folder = sys.argv[4]
    only_gpu = True if sys.argv[5] == 'True' else False

    start = datetime.datetime.now()
    multiple_layers = ((0, 200), (200, 700), (700, 2000), (0, 100),
                       (100, 200), (200, 300), (300, 400),
                       (400, 500), (500, 1000))
    mask, e3t, depth = load_mesh(mesh_file)
    weights = compute_weights(multiple_layers, mask, e3t, depth, only_gpu)
    basin_name, basins = load_basins(regions_file)
    e1t, e2t = load_areas(mesh_file)
    area = compute_areas_basin(e1t, e2t, basins)
    for file in os.listdir(dataset_folder):
        if file.endswith(".nc"):
            thetao = load_thetao(os.path.join(dataset_folder, file))
            all_ohc = compute_OHC(multiple_layers, weights, thetao, area, only_gpu)
            save_data(multiple_layers, basin_name, all_ohc, file, output_folder)
    compss_barrier()
    seconds_total = datetime.datetime.now() - start
    print('Total elapsed time: {0}'.format(seconds_total))


@task(returns=3, mesh_file=FILE_IN)
def load_mesh(mesh_file):
    import iris
    mask = iris.load_cube(mesh_file, 'tmask').data.astype(np.float32)
    e3t = iris.load_cube(mesh_file, 'e3t_0').data.astype(np.float32)
    depth = iris.load_cube(mesh_file, 'gdepw_0').data.astype(np.float32)
    return mask, e3t, depth


@task(returns=1)
def compute_weights(layers, mask, e3t, depth, only_gpu):
    weights = []
    for min_depth, max_depth in layers:
        if only_gpu:
            weights.append(calculate_weight_numba_cuda_version(min_depth,
                                                               max_depth,
                                                               e3t,
                                                               depth,
                                                               mask))
        else:
            weights.append(calculate_weight_numba(min_depth,
                                                  max_depth,
                                                  e3t,
                                                  depth,
                                                  mask))
    return weights


@vectorize(['float32(int32, int32, float32, float32, float32)'], target='cpu')
def calculate_weight_numba(min_depth, max_depth, e3t, depth, mask):
    """
    Calculates the weight for each cell
    """
    if not mask:
        return 0
    top = depth
    bottom = top + e3t
    if bottom < min_depth or top > max_depth:
        return 0
    else:
        if top < min_depth:
            top = min_depth
        if bottom > max_depth:
            bottom = max_depth
        return (bottom - top) * 1020 * 4000


def calculate_weight_numba_cuda_version(min_depth, max_depth,
                                        e3t, depth, mask):
    # The difference is that we load all data per task, instead of only once
    # and compute all layers. We will see if it is worth it.
    gpu_mask = cuda.to_device(mask.data.astype(np.float32))
    gpu_e3t = cuda.to_device(e3t.data.astype(np.float32))
    gpu_depth = cuda.to_device(depth.data.astype(np.float32))
    weight = calculate_weight_numba_cuda(min_depth, max_depth,
                                         gpu_e3t, gpu_depth, gpu_mask)
    local_weight = weight.copy_to_host()
    return local_weight


@vectorize(['float32(int32, int32, float32, float32, float32)'], target='cuda')
def calculate_weight_numba_cuda(min_depth, max_depth, e3t, depth, mask):
    """
    Calculates the weight for each cell
    """
    if not mask:
        return 0
    top = depth
    bottom = top + e3t
    if bottom < min_depth or top > max_depth:
        return 0
    else:
        if top < min_depth:
            top = min_depth
        if bottom > max_depth:
            bottom = max_depth

        return (bottom - top) * 1020 * 4000


@task(returns=2, regions_file=FILE_IN)
def load_basins(regions_file):
    import iris
    regions = iris.load(regions_file)
    basins = {}
    basin_name = []
    for region in regions:
        name = region.name()
        if name in ('nav_lat', 'nav_lon'):
            continue
        basin_name.append(name)
        extract_region = region.extract(iris.Constraint(z=1))
        basins[name] = extract_region.data.astype(np.float32)
    return basin_name, basins


@task(returns=2, mesh_file=FILE_IN)
def load_areas(mesh_file):
    import iris
    e1t = iris.load_cube(mesh_file, 'e1t').data.astype(np.float32)
    e2t = iris.load_cube(mesh_file, 'e2t').data.astype(np.float32)
    return e1t, e2t


@task(returns=1)
def compute_areas_basin(e1t, e2t, basins):
    area = []
    for basin in basins:
        area.append(compute_area_basin(e1t, e2t, basins[basin]))
    return area


@vectorize(['float32(float32, float32, float32)'], target='cpu')
def compute_area_basin(e1t, e2t, basin):
    return e1t*e2t*basin


@task(returns=1, data_file=FILE_IN)
def load_thetao(data_file):
    import iris
    thetao = iris.load_cube(data_file, 'sea_water_potential_temperature')
    thetao_data = np.ma.filled(thetao.data, 0.0).astype(np.float32)
    thetao_data += 273.15
    return thetao_data


@task(returns=1)
def compute_OHC(layers, weights, thetao, area, only_gpu):
    all_ohc = []
    for layer in range(len(layers)):
        if only_gpu:
            all_ohc.append(compute_ohc_gpu(layer, thetao, weights, area))
        else:
            all_ohc.append(compute_ohc_cpu(layer, thetao, weights, area))
    return all_ohc


def compute_ohc_cpu(layer, thetao, weights, area):
    ohc_layer = sum_red.reduce(multiply_thetao_weight(thetao, weights[layer]),
                               axis=1)
    ohc1D_total = []
    for basin in area:
        ohc_basin = multiply_array(ohc_layer, basin)
        temp = sum_red.reduce(ohc_basin, axis=1)
        ohc1D = sum_red.reduce(temp, axis=1)
        ohc1D_total.append(ohc1D)
    return [ohc_layer, ohc1D_total]


def compute_ohc_gpu(layer, thetao, weights, area):
    levels = thetao.shape[1]
    times = thetao.shape[0]
    lats = thetao.shape[2]
    lons = thetao.shape[3]
    basins = len(area)

    area_basin = np.empty((basins, lats, lons))
    block = (128, 1, 1)
    grid_size = (lons // block[0]) + 1
    grid = (grid_size, lats, times)

    gpu_ohc = cuda.device_array((times, lats, lons), dtype=np.float32)
    gpu_temp = cuda.device_array((basins, times, lats*lons), dtype=np.float32)
    for i, basin in enumerate(area):
        area_basin[i, :, :] = basin
    gpu_basins_area = cuda.to_device(area_basin.astype(np.float32))
    gpu_thetao = cuda.to_device(thetao.astype(np.float32))

    compute_ohc[grid, block](gpu_thetao, weights[layer], gpu_ohc, levels)
    ohc = gpu_ohc.copy_to_host()  # moure al final
    multiply_ohc_basin[grid, block](gpu_ohc, gpu_basins_area, gpu_temp)
    ohc1D_basin = []
    for basin in range(basins):
        ohc_1D = np.empty(times, dtype=np.float32)
        for time in range(times):
            ohc_1D[time] = sum_red_cuda(gpu_temp[basin, time, :])
        ohc1D_basin.append(ohc_1D)
    del gpu_ohc, gpu_temp, gpu_basins_area

    return [ohc, ohc1D_basin]


@task()
def save_data(layers, basins, ohc, name, output_folder):
    import iris
    ohc_cube = []
    for i, layer in enumerate(layers):
        ohc_cube.append(iris.cube.Cube(ohc[i][0],
                                       long_name='Ocean heat content'
                                                 ' {0[0]} to {0[1]} meters'
                                       .format(layer)))
        ohc_1D = []
        for j, basin in enumerate(basins):
            ohc_1D.append(iris.cube.Cube(ohc[i][1][j][:],
                          long_name='{0}'.format(basin)))
        iris.save(ohc_1D,
                  output_folder + name + '_ohc_1D_{0}_numba_vec.nc'
                  .format(layer), zlib=True)
    iris.save(ohc_cube,
              output_folder + name + '_ohc_pycompss.nc', zlib=True)


@cuda.jit()
def compute_ohc(thetao, weight, ohc, levels):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    j = cuda.blockIdx.y
    t = cuda.blockIdx.z
    # i, j, t = grid(3)
    ohc[t, j, i] = 0.0
    temp = 0.0
    if(i >= thetao.shape[3]):
        return
    for lev in range(levels):
        thetao_value = thetao[t, lev, j, i]
        temp += thetao_value * weight[0, lev, j, i]
    ohc[t, j, i] = temp


@cuda.jit()
def multiply_ohc_basin(ohc, area, temp):
    i, j, t = cuda.grid(3)
    basins = area.shape[0]
    x = i+j*ohc.shape[2]
    if(i >= ohc.shape[2]):
        return
    for basin in range(basins):
        temp[basin, t, x] = 0.0
        temp[basin, t, x] = ohc[t, j, i] * area[basin, j, i]


@vectorize(['float32(float32, float32)'], target='cpu')
def multiply_thetao_weight(thetao, weight):
    return thetao*weight


@vectorize(['float64(float32, float32)', 'float64(float64, float64)'],
           target='cpu')
def sum_red(x, y):
    return x+y


@vectorize(['float32(float32, float32)', 'float64(float64, float64)'],
           target='cpu')
def multiply_array(a, b):
    return a*b


@cuda.reduce
def sum_red_cuda(x, y):
    return x+y


if __name__ == '__main__':
    if (len(sys.argv) != 6):
        print("ERROR: Wrong number of parameters.")
        usage = "USAGE: oceanheatcontent.py " + \
                "<DATASET_FOLDER> <MESH_FILE> <REGIONS_FILE> " + \
                "<OUTPUT_FOLDER> <ONLY_GPU>"
        print(usage)
    else:
        main()
