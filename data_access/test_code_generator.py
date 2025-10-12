import xarray as xr
import fsspec
from fsspec.implementations.zip import ZipFileSystem
from itertools import product
import pandas as pd
from datetime import datetime


# URL Configuration matching the HTML generator
class URLConfig:
    """URL builder for GLOBGM datasets"""
    
    HISTORICAL_BASE = "https://geo.public.data.uu.nl/vault-globgm-historical-reference-gswp3-w5e5/research-globgm-historical-reference-gswp3-w5e5%5B1754035745%5D/original/"
    CMIP6_MONTHLY = "https://geo.public.data.uu.nl/vault-globgm-cmip6-monthly/research-globgm-cmip6-monthly%5B1755499987%5D/original/"
    CMIP6_ANNUAL = "https://geo.public.data.uu.nl/vault-globgm-cmip6-annual/research-globgm-cmip6-annual%5B1755499806%5D/original/"
    CMIP6_AVERAGE = "https://geo.public.data.uu.nl/vault-globgm-cmip6-average/research-globgm-cmip6-average%5B1755499873%5D/original/"
    
    @staticmethod
    def build_historical_url(variable, temporal):
        extension = "nc" if temporal == "average" else "zarr.zip"
        filename = f"{variable}_reference_gswp3-w5e5_{temporal}_1960_2019.{extension}"
        return f"{URLConfig.HISTORICAL_BASE}{temporal}/{filename}"
    
    @staticmethod
    def build_cmip6_url(variable, temporal, scenario, cmip6_type, gcm_model=None):
        year_ranges = {
            "historical": "1960_2014",
            "ssp126": "2015_2100",
            "ssp370": "2015_2100",
            "ssp585": "2015_2100"
        }
        year_range = year_ranges[scenario]
        
        # Determine base URL and extension
        if temporal == "monthly":
            base_url = URLConfig.CMIP6_MONTHLY
        elif temporal == "annual":
            base_url = URLConfig.CMIP6_ANNUAL
        else:  # average
            base_url = URLConfig.CMIP6_AVERAGE
        
        extension = "nc" if temporal == "average" else "zarr.zip"
        
        # Build filename
        if cmip6_type == "ensemble":
            # Special handling for monthly ensemble files - inconsistent naming
            scenario_in_filename = scenario
            
            if temporal == "monthly":
                if variable == "hds":
                    # hds: all SSP scenarios drop "ssp" prefix
                    if scenario == "ssp126":
                        scenario_in_filename = "126"
                    elif scenario == "ssp370":
                        scenario_in_filename = "370"
                    elif scenario == "ssp585":
                        scenario_in_filename = "585"
                elif variable == "wtd":
                    # wtd: only ssp126 drops "ssp" prefix, others keep it
                    if scenario == "ssp126":
                        scenario_in_filename = "126"
                    # ssp370 and ssp585 keep "ssp" prefix
            
            filename = f"{variable}_{temporal}_{year_range}_{scenario_in_filename}_ensemble.{extension}"
            folder = f"ensemble_{temporal}"
        else:  # gcm
            filename = f"{variable}_{temporal}_{year_range}_{scenario}_{gcm_model}.{extension}"
            folder = f"GCM_{temporal}"
        
        return f"{base_url}{folder}/{filename}"


def test_url_exists(url):
    """Test if URL is accessible"""
    try:
        fs = fsspec.filesystem('http')
        fs.info(url)
        return True, None
    except Exception as e:
        return False, str(e)


def load_dataset(url):
    """Load dataset from either NetCDF or zipped Zarr format"""
    try:
        if url.endswith('.nc'):
            ds = xr.open_dataset(url, chunks={}, engine='h5netcdf')
        else:
            file_object = fsspec.open(url).open()
            zip_fs = ZipFileSystem(file_object, mode='r')
            store_mapper = fsspec.FSMap('/', zip_fs, check=False, create=False)
            ds = xr.open_zarr(store_mapper, zarr_format=2)
        
        # Get basic info
        dims = dict(ds.dims)
        vars = list(ds.data_vars)
        ds.close()
        
        return True, dims, vars, None
    except Exception as e:
        return False, None, None, str(e)


def test_all_combinations():
    """Test all possible combinations of dataset configurations"""
    
    # Define all options
    variables = ["hds", "wtd"]
    temporal_options_all = ["average", "annual", "monthly"]
    temporal_options_gcm = ["average", "annual"]  # No monthly for GCM
    gcm_models = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
    scenarios = ["historical", "ssp126", "ssp370", "ssp585"]
    
    results = []
    
    print("=" * 80)
    print("GLOBGM DATASET TESTER")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Test Historical Reference datasets
    print("\n--- Testing Historical Reference (GSWP3-W5E5) ---")
    for variable, temporal in product(variables, temporal_options_all):
        url = URLConfig.build_historical_url(variable, temporal)
        
        print(f"\nTesting: {variable} - {temporal}")
        print(f"URL: {url}")
        
        # Test URL accessibility
        url_exists, url_error = test_url_exists(url)
        
        if url_exists:
            print("✅ URL accessible")
            # Test data loading
            load_success, dims, vars, load_error = load_dataset(url)
            
            if load_success:
                print(f"✅ Data loaded successfully")
                print(f"   Dimensions: {dims}")
                print(f"   Variables: {vars}")
            else:
                print(f"❌ Failed to load data: {load_error}")
        else:
            print(f"❌ URL not accessible: {url_error}")
            load_success, dims, vars = False, None, None
        
        results.append({
            "dataset_type": "historical",
            "variable": variable,
            "temporal": temporal,
            "scenario": "N/A",
            "cmip6_type": "N/A",
            "gcm_model": "N/A",
            "url": url,
            "url_accessible": url_exists,
            "data_loaded": load_success,
            "dimensions": str(dims) if dims else None,
            "error": url_error or load_error if not (url_exists and load_success) else None
        })
    
    # Test CMIP6 Ensemble datasets (all temporal options)
    print("\n\n--- Testing CMIP6 Ensemble ---")
    for variable, temporal, scenario in product(variables, temporal_options_all, scenarios):
        url = URLConfig.build_cmip6_url(variable, temporal, scenario, "ensemble")
        
        print(f"\nTesting: {variable} - {temporal} - {scenario} - ensemble")
        print(f"URL: {url}")
        
        # Test URL accessibility
        url_exists, url_error = test_url_exists(url)
        
        if url_exists:
            print("✅ URL accessible")
            load_success, dims, vars, load_error = load_dataset(url)
            
            if load_success:
                print(f"✅ Data loaded successfully")
                print(f"   Dimensions: {dims}")
            else:
                print(f"❌ Failed to load data: {load_error}")
        else:
            print(f"❌ URL not accessible: {url_error}")
            load_success, dims, vars = False, None, None
        
        results.append({
            "dataset_type": "cmip6",
            "variable": variable,
            "temporal": temporal,
            "scenario": scenario,
            "cmip6_type": "ensemble",
            "gcm_model": "N/A",
            "url": url,
            "url_accessible": url_exists,
            "data_loaded": load_success,
            "dimensions": str(dims) if dims else None,
            "error": url_error or load_error if not (url_exists and load_success) else None
        })
    
    # Test CMIP6 Individual GCM datasets (NO MONTHLY - only average and annual)
    print("\n\n--- Testing CMIP6 Individual GCM (all models, no monthly) ---")
    for variable, temporal, scenario, gcm_model in product(variables, temporal_options_gcm, scenarios, gcm_models):
        url = URLConfig.build_cmip6_url(variable, temporal, scenario, "gcm", gcm_model)
        
        print(f"\nTesting: {variable} - {temporal} - {scenario} - {gcm_model}")
        print(f"URL: {url}")
        
        # Test URL accessibility
        url_exists, url_error = test_url_exists(url)
        
        if url_exists:
            print("✅ URL accessible")
            load_success, dims, vars, load_error = load_dataset(url)
            
            if load_success:
                print(f"✅ Data loaded successfully")
                print(f"   Dimensions: {dims}")
            else:
                print(f"❌ Failed to load data: {load_error}")
        else:
            print(f"❌ URL not accessible: {url_error}")
            load_success, dims, vars = False, None, None
        
        results.append({
            "dataset_type": "cmip6",
            "variable": variable,
            "temporal": temporal,
            "scenario": scenario,
            "cmip6_type": "gcm",
            "gcm_model": gcm_model,
            "url": url,
            "url_accessible": url_exists,
            "data_loaded": load_success,
            "dimensions": str(dims) if dims else None,
            "error": url_error or load_error if not (url_exists and load_success) else None
        })
    
    # Create summary report
    df = pd.DataFrame(results)
    
    print("\n\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    # Calculate totals
    hist_count = len(df[df['dataset_type'] == 'historical'])
    cmip6_ensemble_count = len(df[(df['dataset_type'] == 'cmip6') & (df['cmip6_type'] == 'ensemble')])
    cmip6_gcm_count = len(df[(df['dataset_type'] == 'cmip6') & (df['cmip6_type'] == 'gcm')])
    
    total_tests = len(df)
    url_success = df['url_accessible'].sum()
    load_success = df['data_loaded'].sum()
    
    print(f"\n📊 Test breakdown:")
    print(f"   Historical: {hist_count} tests (2 vars × 3 temporal)")
    print(f"   CMIP6 Ensemble: {cmip6_ensemble_count} tests (2 vars × 3 temporal × 4 scenarios)")
    print(f"   CMIP6 Individual GCM: {cmip6_gcm_count} tests (2 vars × 2 temporal × 4 scenarios × 5 GCMs)")
    print(f"   → Note: Monthly excluded for Individual GCM")
    print(f"   → Note: Monthly ensemble naming inconsistent (hds drops 'ssp', wtd keeps ssp370/585)")
    
    print(f"\n📈 Results:")
    print(f"   Total combinations tested: {total_tests}")
    print(f"   URLs accessible: {url_success}/{total_tests} ({url_success/total_tests*100:.1f}%)")
    print(f"   Data loaded successfully: {load_success}/{total_tests} ({load_success/total_tests*100:.1f}%)")
    
    # Breakdown by category
    print(f"\n📂 Success by category:")
    for dataset_type in df['dataset_type'].unique():
        if dataset_type == 'historical':
            subset = df[df['dataset_type'] == dataset_type]
            success = subset['data_loaded'].sum()
            print(f"   Historical: {success}/{len(subset)} ({success/len(subset)*100:.1f}%)")
        else:
            for cmip6_type in df[df['dataset_type'] == 'cmip6']['cmip6_type'].unique():
                subset = df[(df['dataset_type'] == 'cmip6') & (df['cmip6_type'] == cmip6_type)]
                success = subset['data_loaded'].sum()
                print(f"   CMIP6 {cmip6_type.capitalize()}: {success}/{len(subset)} ({success/len(subset)*100:.1f}%)")
    
    # Show failures
    failures = df[~df['data_loaded']]
    if len(failures) > 0:
        print(f"\n❌ Failed tests: {len(failures)}")
        print("\nFailed combinations:")
        for idx, row in failures.iterrows():
            print(f"\n  - {row['variable']} | {row['temporal']} | {row['scenario']} | {row['cmip6_type']} | {row['gcm_model']}")
            print(f"    URL: {row['url']}")
            if not row['url_accessible']:
                print(f"    ⚠️  URL not accessible: {row['error']}")
            else:
                print(f"    ⚠️  Data loading failed: {row['error']}")
    else:
        print("\n✅ All tests passed!")
    
    # Save results to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"globgm_test_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return df


if __name__ == "__main__":
    results_df = test_all_combinations()
    
    # Optional: Show summary statistics
    print("\n" + "=" * 80)
    print("DETAILED STATISTICS")
    print("=" * 80)
    
    # Group by temporal aggregation
    print("\n📊 Success rate by temporal aggregation:")
    temporal_stats = results_df.groupby('temporal')['data_loaded'].agg(['sum', 'count', lambda x: (x.sum()/len(x)*100)])
    temporal_stats.columns = ['Success', 'Total', 'Percentage']
    print(temporal_stats.to_string())
    
    # Group by variable
    print("\n📊 Success rate by variable:")
    variable_stats = results_df.groupby('variable')['data_loaded'].agg(['sum', 'count', lambda x: (x.sum()/len(x)*100)])
    variable_stats.columns = ['Success', 'Total', 'Percentage']
    print(variable_stats.to_string())
    
    # For CMIP6, group by scenario
    cmip6_data = results_df[results_df['dataset_type'] == 'cmip6']
    if len(cmip6_data) > 0:
        print("\n📊 CMIP6 success rate by scenario:")
        scenario_stats = cmip6_data.groupby('scenario')['data_loaded'].agg(['sum', 'count', lambda x: (x.sum()/len(x)*100)])
        scenario_stats.columns = ['Success', 'Total', 'Percentage']
        print(scenario_stats.to_string())
