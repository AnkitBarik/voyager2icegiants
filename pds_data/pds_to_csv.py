#!/usr/bin/env python3
"""Convert Voyager 2 MAG PDS ASCII tables (*.TAB) to headed CSV files.

Handles both label flavours found in these products:

  * PDS3 -- column names come from the ASCDATA.FMT structure file that sits
    alongside the tables (one .FMT describes every .TAB in the folder).
  * PDS4 -- column names come from a per-table .lblx XML label, which also
    names the .TAB file it describes.

Usage:
    python pds_to_csv.py [folder ...]     # default: every known data folder
"""

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PDS4_NS = {"p": "http://pds.nasa.gov/pds4/pds/v1"}

# Fill handling. The labels declare missing_constant 999.999, but the tables
# actually write 999.000; the PDS4 labels additionally declare 90.0 for DELTA
# and 45.0 for LAMBDA, which are the values arcsin/atan return when fed the
# fill field and are otherwise perfectly valid angles. Rather than trust any
# of that, gaps are identified by NPTS == 0 (no samples in the averaging
# interval), which matches BMAG == 999.0 exactly in every table here.
FILL_VALUES = (999.999, 999.000)


def parse_pds3_fmt(fmt_path):
    """Columns from a PDS3 .FMT structure file."""
    fields, current = [], None
    for line in fmt_path.read_text().splitlines():
        line = line.strip()
        if re.match(r"OBJECT\s*=\s*COLUMN$", line):
            current = {}
            continue
        if re.match(r"END_OBJECT\s*=\s*COLUMN$", line):
            if current:
                fields.append(current)
            current = None
            continue
        if current is None:
            continue
        m = re.match(r"([A-Z_]+)\s*=\s*(.*)$", line)
        if m:
            current.setdefault(m.group(1), m.group(2).strip().strip('"'))
    return [
        {
            "name": f["NAME"],
            "is_text": f["DATA_TYPE"] == "CHARACTER",
            "missing": float(f["MISSING_CONSTANT"]) if "MISSING_CONSTANT" in f else None,
        }
        for f in fields
    ]


def parse_pds4_lblx(lblx_path):
    """(table filename, columns) from a PDS4 observational label."""
    root = ET.parse(lblx_path).getroot()
    area = root.find(".//p:File_Area_Observational", PDS4_NS)
    if area is None:
        return None, None  # e.g. a collection inventory label
    table_name = area.findtext(".//p:file_name", namespaces=PDS4_NS)

    fields = []
    for fc in area.findall(".//p:Field_Character", PDS4_NS):
        miss = fc.findtext(".//p:missing_constant", namespaces=PDS4_NS)
        dtype = fc.findtext("p:data_type", namespaces=PDS4_NS) or ""
        fields.append(
            {
                "name": fc.findtext("p:name", namespaces=PDS4_NS),
                "is_text": dtype in ("ASCII_String", "ASCII_Date_Time_YMD_UTC"),
                "missing": float(miss) if miss is not None else None,
            }
        )
    return table_name, fields


def convert_table(tab_path, fields):
    names = [f["name"] for f in fields]
    df = pd.read_csv(tab_path, header=None, names=names, skipinitialspace=True)

    for f in fields:
        if f["is_text"]:
            df[f["name"]] = df[f["name"]].astype(str).str.strip()

    # PDSTIME carries a trailing 'Z' (UTC) -> parse to a real timestamp.
    if "PDSTIME" in df:
        df["PDSTIME"] = pd.to_datetime(
            df["PDSTIME"].str.rstrip("Z"), format="%Y-%m-%dT%H:%M:%S.%f", utc=True
        )

    # Gap rows: no samples averaged, so every measurement column is fill.
    gap = df["NPTS"] == 0 if "NPTS" in df else pd.Series(False, index=df.index)

    notes = []
    for f in fields:
        if f["missing"] is None or f["is_text"]:
            continue
        col = df[f["name"]]
        mask = gap | np.logical_or.reduce([np.isclose(col, v) for v in FILL_VALUES])
        if mask.any():
            df.loc[mask, f["name"]] = pd.NA
            notes.append(f"{f['name']}: {int(mask.sum())} fill -> NaN")
    if gap.any():
        notes.append(f"({int(gap.sum())} gap rows flagged by NPTS == 0)")

    out_path = tab_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"  {tab_path.name} -> {out_path.name}  ({len(df)} rows x {len(df.columns)} cols)")
    for n in notes:
        print(f"      ! {n}")
    return df


def convert_folder(folder):
    folder = Path(folder)
    if not folder.is_absolute():
        folder = HERE / folder
    print(f"{folder.relative_to(HERE) if folder.is_relative_to(HERE) else folder}")

    lblx_files = sorted(p for p in folder.glob("*.lblx") if not p.name.startswith("collection"))
    if lblx_files:  # PDS4
        for lblx in lblx_files:
            table_name, fields = parse_pds4_lblx(lblx)
            if table_name is None:
                continue
            convert_table(folder / table_name, fields)
        return

    fmt_path = folder / "ASCDATA.FMT"
    if fmt_path.exists():  # PDS3
        fields = parse_pds3_fmt(fmt_path)
        for tab in sorted(folder.glob("*.TAB")):
            convert_table(tab, fields)
        return

    print("  (no PDS3 .FMT or PDS4 .lblx label found)")


def discover():
    """Every folder under this script that holds a .TAB plus a label."""
    out = []
    for tab in sorted(HERE.rglob("*.TAB")):
        d = tab.parent
        if d in out:
            continue
        if (d / "ASCDATA.FMT").exists() or any(d.glob("*.lblx")):
            out.append(d)
    return out


if __name__ == "__main__":
    targets = [Path(a) for a in sys.argv[1:]] or discover()
    for t in targets:
        convert_folder(t)
