"""Flat file manifest for the rclone download generator.

The generator must never *construct* a filename. Deposited names are not fully
regular - the monthly ensemble product drops the "ssp" prefix for hds in every
scenario, and for wtd only in ssp126 - so any client-side filename builder
encodes those exceptions as special cases and drifts the moment a file is
re-deposited. Instead we read the paths that are actually in the vault and let
the UI filter them.

Vault roots and the CSV manifests are owned by ``data_catalogue/_catalogue.py``.
This module imports them rather than restating them.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# The catalogue package is a sibling directory, not an installed module.
_CATALOGUE_DIR = Path(__file__).resolve().parent.parent / "data_catalogue"
if str(_CATALOGUE_DIR) not in sys.path:
    sys.path.insert(0, str(_CATALOGUE_DIR))

import _catalogue as cat  # noqa: E402

# --- Display labels ---------------------------------------------------------

PRODUCT_LABELS = {
    "reference": "Historical reference (GSWP3-W5E5)",
    "ensemble": "CMIP6 ensemble",
    "gcm": "CMIP6 individual GCM",
}

VARIABLE_LABELS = {
    "hds": "Hydraulic head",
    "wtd": "Water table depth",
}

AGGREGATION_LABELS = {
    "average": "Long-term average",
    "annual": "Annual",
    "monthly": "Monthly",
}

SCENARIO_LABELS = {
    "historical": "Historical",
    "ssp126": "SSP1-2.6",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

GCM_LABELS = {
    "gfdl-esm4": "GFDL-ESM4",
    "ipsl-cm6a-lr": "IPSL-CM6A-LR",
    "mpi-esm1-2-hr": "MPI-ESM1-2-HR",
    "mri-esm2-0": "MRI-ESM2-0",
    "ukesm1-0-ll": "UKESM1-0-LL",
}

# Order matters: these drive the order checkboxes appear in the form.
PRODUCT_ORDER = ["reference", "ensemble", "gcm"]
VARIABLE_ORDER = ["hds", "wtd"]
AGGREGATION_ORDER = ["average", "annual", "monthly"]
SCENARIO_ORDER = ["historical", "ssp126", "ssp370", "ssp585"]
GCM_ORDER = list(GCM_LABELS)

# --- Manifest sources -------------------------------------------------------

# (csv name, vault key, product facet)
_SOURCES = [
    ("historical_reference.csv", "reference", "reference"),
    ("cmip6_average_ensemble.csv", "average", "ensemble"),
    ("cmip6_annual_ensemble.csv", "annual", "ensemble"),
    ("cmip6_monthly_ensemble.csv", "monthly", "ensemble"),
    ("cmip6_average_GCM.csv", "average", "gcm"),
    ("cmip6_annual_GCM.csv", "annual", "gcm"),
]

# hds_reference_gswp3-w5e5_annual_1960_2019.zarr.zip
_REFERENCE_RE = re.compile(
    r"^(?P<var>hds|wtd)_reference_gswp3-w5e5_"
    r"(?P<agg>average|annual|monthly)_"
    r"(?P<start>\d{4})_(?P<end>\d{4})\.(?P<ext>nc|zarr\.zip)$"
)

# hds_annual_2015_2100_ssp126_ensemble.zarr.zip
# hds_monthly_2015_2100_126_ensemble.zarr.zip   <- note the bare scenario token
# wtd_annual_2015_2100_ssp585_ukesm1-0-ll.zarr.zip
_CMIP6_RE = re.compile(
    r"^(?P<var>hds|wtd)_"
    r"(?P<agg>average|annual|monthly)_"
    r"(?P<start>\d{4})_(?P<end>\d{4})_"
    r"(?P<scen>historical|ssp\d{3}|\d{3})_"
    r"(?P<member>.+?)\.(?P<ext>nc|zarr\.zip)$"
)


def _normalise_scenario(token: str) -> str:
    """``126`` and ``ssp126`` are the same scenario, deposited inconsistently."""
    return f"ssp{token}" if token.isdigit() else token


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def load_files() -> list[dict]:
    """Every downloadable file, as flat records the form can filter."""
    records: list[dict] = []

    for csv_name, vault_key, product in _SOURCES:
        raw = pd.read_csv(
            _CATALOGUE_DIR / csv_name,
            header=None,
            names=["path", "size", "sha256"],
            skip_blank_lines=True,
        )
        raw["path"] = raw["path"].str.strip()

        for row in raw.itertuples():
            name = _basename(row.path)

            if product == "reference":
                m = _REFERENCE_RE.match(name)
                if m is None:
                    raise ValueError(f"{csv_name}: cannot parse {name!r}")
                scenario, gcm = "historical", None
            else:
                m = _CMIP6_RE.match(name)
                if m is None:
                    raise ValueError(f"{csv_name}: cannot parse {name!r}")
                scenario = _normalise_scenario(m["scen"])
                member = m["member"]
                if product == "ensemble":
                    if member != "ensemble":
                        raise ValueError(f"{csv_name}: expected ensemble, got {member!r}")
                    gcm = None
                else:
                    if member == "ensemble":
                        raise ValueError(f"{csv_name}: unexpected ensemble member in GCM manifest")
                    gcm = member

            records.append(
                {
                    "product": product,
                    "variable": m["var"],
                    "aggregation": m["agg"],
                    "scenario": scenario,
                    "gcm": gcm,
                    "period": f"{m['start']}-{m['end']}",
                    "format": "NetCDF" if m["ext"] == "nc" else "Zarr (zipped)",
                    "filename": name,
                    # The subfolder the file sits in, relative to the vault root.
                    "subfolder": row.path.rsplit("/", 1)[0] if "/" in row.path else "",
                    "vault": vault_key,
                    "vault_root": cat.YODA_VAULTS[vault_key],
                    "url": cat.YODA_VAULTS[vault_key] + row.path,
                    "size_gib": cat.parse_size(row.size),
                }
            )

    return records


def facet_options(records: list[dict]) -> dict:
    """Label maps and display order, so the JS never hardcodes vocabulary."""
    return {
        "product": [
            {"value": v, "label": PRODUCT_LABELS[v]}
            for v in PRODUCT_ORDER
            if any(r["product"] == v for r in records)
        ],
        "variable": [
            {"value": v, "label": VARIABLE_LABELS[v]}
            for v in VARIABLE_ORDER
            if any(r["variable"] == v for r in records)
        ],
        "aggregation": [
            {"value": v, "label": AGGREGATION_LABELS[v]}
            for v in AGGREGATION_ORDER
            if any(r["aggregation"] == v for r in records)
        ],
        "scenario": [
            {"value": v, "label": SCENARIO_LABELS[v]}
            for v in SCENARIO_ORDER
            if any(r["scenario"] == v for r in records)
        ],
        "gcm": [
            {"value": v, "label": GCM_LABELS[v]}
            for v in GCM_ORDER
            if any(r["gcm"] == v for r in records)
        ],
    }
