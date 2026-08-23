"""Parsing and rendering helpers for the GLOBGM CMIP6 data catalogue.

The seven CSVs in this directory are manifests of what was deposited in the YODA
vaults: one row per file, ``path,size,sha256``, no header. The filenames encode
variable, aggregation, period, scenario and ensemble member; this module turns
that string into real columns so the catalogue page can be organised by what the
data *is* rather than by which vault it landed in.

Parsing is strict on purpose. A filename that does not match raises rather than
producing NaN columns, because a silent parse failure becomes a wrong download
link.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import itables
from itables import show

# ``searching`` and ``info`` are valid DataTables options that itables does not
# list in its own typed schema; without this the SyntaxWarning is captured by
# Quarto and rendered as visible output on the page.
itables.options.warn_on_undocumented_option = False

HERE = Path(__file__).parent

# --- Vaults -----------------------------------------------------------------
# Square brackets are percent-encoded in every entry, including quality, which
# the old inline chunks left raw.

_BASE = "https://geo.public.data.uu.nl"

YODA_VAULTS = {
    "reference": f"{_BASE}/vault-globgm-historical-reference-gswp3-w5e5/research-globgm-historical-reference-gswp3-w5e5%5B1754035745%5D/original/",
    "monthly": f"{_BASE}/vault-globgm-cmip6-monthly/research-globgm-cmip6-monthly%5B1755499987%5D/original/",
    "annual": f"{_BASE}/vault-globgm-cmip6-annual/research-globgm-cmip6-annual%5B1755499806%5D/original/",
    "average": f"{_BASE}/vault-globgm-cmip6-average/research-globgm-cmip6-average%5B1755499873%5D/original/",
    "quality": f"{_BASE}/vault-globgm-cmip6-quality/research-globgm-cmip6-quality%5B1755500055%5D/original/",
}

DOI_LANDING = {
    "reference": "https://public.yoda.uu.nl/geo/UU01/AKSHOX.html",
    "monthly": "https://public.yoda.uu.nl/geo/UU01/1BXLPD.html",
    "annual": "https://public.yoda.uu.nl/geo/UU01/V6B9YS.html",
    "average": "https://public.yoda.uu.nl/geo/UU01/SLRFI7.html",
    "quality": "https://public.yoda.uu.nl/geo/UU01/16EJ3Y.html",
}

# --- Display maps -----------------------------------------------------------

VARIABLE = {"hds": "Head", "wtd": "Water table depth"}

# Never render a bare "historical": it collides with the observation-forced
# historical reference collection, which is the exact ambiguity this page exists
# to remove.
SCENARIO = {
    "historical": "CMIP6 historical",
    "ssp126": "SSP1-2.6",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}

MEMBER = {
    "ensemble": "Ensemble",
    "gfdl-esm4": "GFDL-ESM4",
    "ipsl-cm6a-lr": "IPSL-CM6A-LR",
    "mpi-esm1-2-hr": "MPI-ESM1-2-HR",
    "mri-esm2-0": "MRI-ESM2-0",
    "ukesm1-0-ll": "UKESM1-0-LL",
}

AGGREGATION = {"average": "Average", "annual": "Annual", "monthly": "Monthly"}

# Coarse to fine, which is also cheapest to heaviest.
AGGREGATION_ORDER = ["Average", "Annual", "Monthly"]

SCENARIO_ORDER = ["CMIP6 historical", "SSP1-2.6", "SSP3-7.0", "SSP5-8.5"]

MEMBER_ORDER = ["Ensemble"] + [MEMBER[k] for k in MEMBER if k != "ensemble"]

# Which CSV belongs to which collection and vault. The parser is told this; it
# must never infer it from the directory prefix, which is inconsistent between
# collections (bare ``annual/`` for the reference, ``ensemble_annual/`` and
# ``GCM_annual/`` for the scenarios).
SCENARIO_SOURCES = [
    ("cmip6_average_ensemble.csv", "average"),
    ("cmip6_average_GCM.csv", "average"),
    ("cmip6_annual_ensemble.csv", "annual"),
    ("cmip6_annual_GCM.csv", "annual"),
    ("cmip6_monthly_ensemble.csv", "monthly"),
]

# --- Filename grammar -------------------------------------------------------

_EXT = r"(?P<ext>\.nc|\.zarr\.zip)"

_REFERENCE_RE = re.compile(
    r"^(?P<var>hds|wtd)_reference_(?P<forcing>gswp3-w5e5)"
    r"_(?P<agg>average|annual|monthly)"
    r"_(?P<start>\d{4})_(?P<end>\d{4})" + _EXT + r"$"
)

_SCENARIO_RE = re.compile(
    r"^(?P<var>hds|wtd)_(?P<agg>average|annual|monthly)"
    r"_(?P<start>\d{4})_(?P<end>\d{4})"
    r"_(?P<scenario>[a-z0-9]+)_(?P<member>[a-z0-9.\-]+)" + _EXT + r"$"
)


class CatalogueParseError(ValueError):
    """A manifest row that the filename grammar does not recognise."""


def parse_size(size: str) -> float:
    """Return a size string such as ``"47.84 GiB"`` in GiB.

    MiB divides by 1024. The inline chunk this replaces divided by 1000, which
    inflated every quality-assurance size by 2.4%.
    """
    value, unit = size.strip().split()
    value = float(value)
    if unit == "GiB":
        return value
    if unit == "MiB":
        return value / 1024
    if unit == "KiB":
        return value / (1024 * 1024)
    if unit == "TiB":
        return value * 1024
    raise CatalogueParseError(f"unrecognised size unit in {size!r}")


def parse_scenario_token(token: str) -> str:
    """Normalise the bare scenario tokens used by four monthly ensemble files.

    ``126`` -> ``ssp126``. Anything already prefixed passes through untouched.
    """
    if token.isdigit():
        return f"ssp{token}"
    return token


def _read_manifest(name: str) -> pd.DataFrame:
    """Read one CSV. Hash fields carry a trailing space in every file."""
    df = pd.read_csv(
        HERE / name,
        header=None,
        names=["path", "size", "sha256"],
        skip_blank_lines=True,
    )
    df["path"] = df["path"].str.strip()
    df["sha256"] = df["sha256"].fillna("").str.strip()
    df["source"] = name
    return df


def _basename(path: str) -> str:
    return path.rsplit("/", 1)[-1]


def _match(regex: re.Pattern, path: str, source: str) -> re.Match:
    m = regex.match(_basename(path))
    if m is None:
        raise CatalogueParseError(f"{source}: cannot parse filename {path!r}")
    return m


def _download_url(vault: str, path: str) -> str:
    """Build a download URL from the *raw* manifest path.

    Normalisation happens only in derived display columns; the URL must use the
    filename exactly as deposited or it will 404.
    """
    return YODA_VAULTS[vault] + path


def load_reference() -> pd.DataFrame:
    """The six observation-forced (GSWP3-W5E5) files."""
    raw = _read_manifest("historical_reference.csv")
    rows = []
    for row in raw.itertuples():
        m = _match(_REFERENCE_RE, row.path, row.source)
        rows.append(
            {
                "Aggregation": AGGREGATION[m["agg"]],
                "Period": f"{m['start']}–{m['end']}",
                "Variable": VARIABLE[m["var"]],
                "Size (GiB)": parse_size(row.size),
                "Download": _download_url("reference", row.path),
            }
        )
    df = pd.DataFrame(rows)
    df["Aggregation"] = pd.Categorical(
        df["Aggregation"], categories=AGGREGATION_ORDER, ordered=True
    )
    return df.sort_values(["Aggregation", "Variable"]).reset_index(drop=True)


def load_scenarios() -> pd.DataFrame:
    """The 104 GCM-forced files, from five manifests, as one table."""
    rows = []
    for name, vault in SCENARIO_SOURCES:
        raw = _read_manifest(name)
        for row in raw.itertuples():
            m = _match(_SCENARIO_RE, row.path, row.source)
            scenario = parse_scenario_token(m["scenario"])
            if scenario not in SCENARIO:
                raise CatalogueParseError(
                    f"{row.source}: unknown scenario {scenario!r} in {row.path!r}"
                )
            if m["member"] not in MEMBER:
                raise CatalogueParseError(
                    f"{row.source}: unknown member {m['member']!r} in {row.path!r}"
                )
            rows.append(
                {
                    "Member": MEMBER[m["member"]],
                    "Period": f"{m['start']}–{m['end']}",
                    "Scenario": SCENARIO[scenario],
                    "Aggregation": AGGREGATION[m["agg"]],
                    "Variable": VARIABLE[m["var"]],
                    "Size (GiB)": parse_size(row.size),
                    "Download": _download_url(vault, row.path),
                }
            )
    df = pd.DataFrame(rows)
    for col, order in (
        ("Aggregation", AGGREGATION_ORDER),
        ("Scenario", SCENARIO_ORDER),
        ("Member", MEMBER_ORDER),
    ):
        df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df.sort_values(
        ["Member", "Scenario", "Aggregation", "Variable"]
    ).reset_index(drop=True)


# What each file in the quality collection holds. Keyed by filename so an
# undescribed deposit raises rather than rendering a blank row.
QUALITY_ITEMS = {
    "quality_assurance.nc": (
        "Static quality flags",
        "Karst aquifers, mountainous regions and permafrost, each as a separate variable",
    ),
    "spinup_info.nc": (
        "Spin-up completion",
        "Month (YYYYMM) at which spin-up is achieved for each cell",
    ),
    "grace_agreement.nc": (
        "GRACE agreement",
        "Categorical classes recording where GRACE agrees with the modelled storage change",
    ),
}

# Deposited in the same manifest but documented elsewhere, so it is skipped
# here rather than tripping the undescribed-file check below.
QUALITY_EXCLUDED = {"ml_bias_correction.nc"}


def load_quality() -> pd.DataFrame:
    """The quality-assurance collection, from ``quality_data.csv``.

    Unlike the other manifests this one carries a header row and a bare
    filename with no directory prefix. The files are not yet deposited, so the
    download URLs it builds do not resolve.
    """
    raw = pd.read_csv(HERE / "quality_data.csv")
    raw.columns = [c.strip() for c in raw.columns]
    rows = []
    for row in raw.itertuples():
        name = row.filename.strip()
        if name in QUALITY_EXCLUDED:
            continue
        if name not in QUALITY_ITEMS:
            raise CatalogueParseError(f"quality_data.csv: undescribed file {name!r}")
        item, contents = QUALITY_ITEMS[name]
        rows.append(
            {
                "Item": item,
                "Contents": contents,
                # MiB, not GiB: these layers run from 131 KiB to 897 MiB, so
                # in GiB the smallest would round to 0.00.
                "Size (MiB)": round(parse_size(row.size) * 1024, 2),
                "Download": _download_url("quality", name),
            }
        )
    order = list(QUALITY_ITEMS.values())
    df = pd.DataFrame(rows)
    df["Item"] = pd.Categorical(
        df["Item"], categories=[i[0] for i in order], ordered=True
    )
    return df.sort_values("Item").reset_index(drop=True)


# --- Rendering --------------------------------------------------------------


def _link(url: str) -> str:
    return f"<a href='{url}' target='_blank' rel='noopener'>Download</a>"


def render_table(df: pd.DataFrame, **kwargs) -> None:
    """One wrapper around ``itables.show`` so table options live in one place.

    Passes a plain DataFrame rather than a Styler: a Styler forces
    ``use_to_html=True``, which materialises every row as HTML and rejects
    ``maxBytes`` outright. The native path hands DataTables a real data array,
    so sorting on ``Size (GiB)`` stays numeric -- but it *does* downsample past
    ``maxBytes``, which defaults to 64KB and would silently truncate the
    104-row scenario table. Hence ``maxBytes=0``.
    """
    display_df = df.copy()
    if "Download" in display_df:
        display_df["Download"] = display_df["Download"].map(
            lambda v: _link(v) if isinstance(v, str) and v.startswith("http") else v
        )
    if "Size (GiB)" in display_df:
        display_df["Size (GiB)"] = display_df["Size (GiB)"].map(
            lambda v: round(v, 2) if isinstance(v, (int, float)) else v
        )
    options = {
        "classes": "display compact",
        "style": "table-layout:auto;width:100%;margin:auto",
        "maxBytes": 0,
        "allow_html": True,
        "showIndex": False,
        "columnDefs": [{"className": "dt-left", "targets": "_all"}],
    }
    options.update(kwargs)
    show(display_df, **options)


def format_size(gib: float) -> str:
    """Human-readable total: TiB above 1024 GiB, GiB below."""
    if gib >= 1024:
        return f"{gib / 1024:.2f} TiB"
    return f"{gib:.1f} GiB"
