import xarray as xr
import fsspec
from fsspec.implementations.zip import ZipFileSystem
from pathlib import Path
from dask.diagnostics import ProgressBar

def load_dataset(url: str) -> xr.Dataset:
    file_object = fsspec.open(url).open()
    zip_fs = ZipFileSystem(file_object, mode='r')
    store_mapper = fsspec.FSMap('/', zip_fs, check=False, create=False)
    return xr.open_zarr(store_mapper, zarr_format=2)

url = "https://geo.public.data.uu.nl/vault-globgm-historical-reference-gswp3-w5e5/research-globgm-historical-reference-gswp3-w5e5%5B1754035745%5D/original/annual/hds_reference_gswp3-w5e5_annual_1960_2019.zarr.zip"
save_path = Path("local_path_to_save_data")

ds = load_dataset(url)
lat_min, lat_max = -34.36, -33.56
lon_min, lon_max = 18.29, 18.96
with ProgressBar():
    so_cape_town = ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max)).compute()
# ds.to_netcdf(save_path / 'hds_reference_gswp3-w5e5_annual_1960_2019.nc')
# ds.to_zarr(save_path / 'hds_reference_gswp3-w5e5_annual_1960_2019.zarr')