"""
This script generates the pictures of a particular variable
for the given nmmb output file.

Variables:
---------
acprec
alwtoa
dust_aod_550
dust_aod_550_b1
dust_aod_550_b2
dust_aod_550_b3
dust_aod_550_b4
dust_aod_550_b5
dust_aod_550_b6
dust_aod_550_b7
dust_aod_550_b8
dust_drydep
dust_load
dust_load_b1
dust_load_b2
dust_load_b3
dust_load_b4
dust_load_b5
dust_load_b6
dust_load_b7
dust_load_b8
dust_pm10_sconc10
dust_pm25_sconc10
dust_sconc
dust_sconc02
dust_sconc10
dust_sconc_b1
dust_sconc_b2
dust_sconc_b3
dust_sconc_b4
dust_sconc_b5
dust_sconc_b6
dust_sconc_b7
dust_sconc_b8
dust_wetdep
dust_wetdep_cuprec
fis
ps
slp
u10
v10
"""
from pycompss.api.task import task
from pycompss.api.parameter import *


@task(FNAME=FILE_IN,
      i1=FILE_OUT, i2=FILE_OUT, i3=FILE_OUT, i4=FILE_OUT, i5=FILE_OUT,
      i6=FILE_OUT, i7=FILE_OUT, i8=FILE_OUT, i9=FILE_OUT)
def generate_figures(FNAME, VNAME, i1, i2, i3, i4, i5, i6, i7, i8, i9):
    """
    Generates the figures for the given variable.
    :param FNAME: Source file
    :param VNAME: Variable
    :param i1: Image 1
    :param i2: Image 2
    :param i3: Image 3
    :param i4: Image 4
    :param i5: Image 5
    :param i6: Image 6
    :param i7: Image 7
    :param i8: Image 8
    :param i9: Image 9
    :return:
    """
    import matplotlib as mpl
    mpl.use('Agg')  # GTK')

    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    from netCDF4 import Dataset as nc
    import numpy as np

    DTYPE = 'float32'

    BOUNDS = [5, 20, 50, 200, 500, 2000, 5000, 20000]
    levels = len(BOUNDS) - 1
    cmap = plt.cm.get_cmap('afmhot_r', levels)
    norm = mpl.colors.BoundaryNorm(BOUNDS, levels)
    cmap.set_under((0, 0, 0, 0))
    cmap.set_over('k')

    f = nc(FNAME)

    # print("Reading var", VNAME, "... ")
    obj = f.variables[VNAME]
    var = obj[:].astype(DTYPE)
    nam = obj.long_name

    # print("Reading coords ... ")
    lon = f.variables['lon'][:].astype(DTYPE)
    lat = f.variables['lat'][:].astype(DTYPE)

    tim = f.variables['time']
    val = tim[:]
    unt = tim.units

    output = [i1, i2, i3, i4, i5, i6, i7, i8, i9]

    o = 0
    for i in np.arange(var.shape[0]):
        # print("Creating plot %s ... " % i)
        plt.clf()
        step = val[i]
        m = Basemap()
        x, y = m(lon, lat)
        m.contourf(x, y, var[i]*10e9, levels=BOUNDS, bounds=BOUNDS, cmap=cmap, norm=norm, extend='both')
        m.drawcoastlines(linewidth='.2')
        m.colorbar(ticks=BOUNDS, size='3%')
        plt.title("NMMB %s $(\mu g/m^3)$\nRun: %s - Fcst: +%02dH" % (nam, unt.split()[2]+' '+unt.split()[3], int(step)))
        # plt.savefig("%sNMMB-%s-%02d.png" % (folder, VNAME, step), dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.savefig(output[o], dpi=200, bbox_inches='tight', pad_inches=0.2)
        o+=1
