load ANACONDA/5.0.1 (PATH, MANPATH, LD_LIBRARY_PATH, LIBRARY_PATH,
PKG_CONFIG_PATH, C_INCLUDE_PATH) 
Traceback (most recent call last):
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/file_manager.py", line 199, in _acquire_with_cache_info
    file = self._cache[self._key]
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/lru_cache.py", line 53, in __getitem__
    value = self._cache[key]
KeyError: [<class 'netCDF4._netCDF4.Dataset'>, ('/gpfs/scratch/bsc21/bsc21005/Eflows4hpc/Ucis4eq/event_ckb_20230908_SAMOS-IZMIR_IRIS_seisensman/salvus_post_swarm/ground_motion_params.nc',), 'r', (('clobber', True), ('diskless', False), ('format', 'NETCDF4'), ('persist', False))]

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/gpfs/projects/bsc44/earthquake/UCIS4EQ/salvus_urgent_simulations_setup/salvus_urgent_wrapper/salvus_plot/plot_outputs.py", line 47, in <module>
    engine='netcdf4'
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/api.py", line 501, in open_dataset
    **kwargs,
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/netCDF4_.py", line 560, in open_dataset
    autoclose=autoclose,
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/netCDF4_.py", line 380, in open
    return cls(manager, group=group, mode=mode, lock=lock, autoclose=autoclose)
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/netCDF4_.py", line 328, in __init__
    self.format = self.ds.data_model
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/netCDF4_.py", line 389, in ds
    return self._acquire()
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/netCDF4_.py", line 383, in _acquire
    with self._manager.acquire_context(needs_lock) as root:
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/contextlib.py", line 112, in __enter__
    return next(self.gen)
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/file_manager.py", line 187, in acquire_context
    file, cached = self._acquire_with_cache_info(needs_lock)
  File "/apps/ANACONDA/5.0.1/envs/salvus/lib/python3.7/site-packages/xarray/backends/file_manager.py", line 205, in _acquire_with_cache_info
    file = self._opener(*self._args, **kwargs)
  File "src/netCDF4/_netCDF4.pyx", line 2307, in netCDF4._netCDF4.Dataset.__init__
  File "src/netCDF4/_netCDF4.pyx", line 1925, in netCDF4._netCDF4._ensure_nc_success
FileNotFoundError: [Errno 2] No such file or directory: b'/gpfs/scratch/bsc21/bsc21005/Eflows4hpc/Ucis4eq/event_ckb_20230908_SAMOS-IZMIR_IRIS_seisensman/salvus_post_swarm/ground_motion_params.nc'
tar: event_ckb_20230908_SAMOS-IZMIR_IRIS_seisensman/salvus_post_swarm/*.png: Cannot stat: No such file or directory
tar: Exiting with failure status due to previous errors
