"""Access to the per-task tables that carry Psych-101's canonical choice codes.

Psych-101 transcripts randomize each participant's response keys, so counting
choices across participants in transcript label space is noise.  The per-task
tables published alongside the dataset do not: they record a canonical
`choice` column that means the same thing for everyone, which is what makes
population-level baselines possible at all.

38 of the test split's experiments have such a table; the rest do not, and no
population baseline exists for them.
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

API = "https://huggingface.co/api/datasets/marcelbinz"
RESOLVE = "https://huggingface.co/datasets/marcelbinz"

FAMILIES = (
    "badham2017deficits", "bahrami2020four", "collsioo2023is",
    "enkavi2019adaptivenback", "enkavi2019digitspan", "enkavi2019gonogo",
    "enkavi2019recentprobes", "feng2021dynamics", "flesch2018comparing",
    "frey2017cct", "frey2017risk", "gershman2018deconstructing",
    "gershman2020reward", "hebart2023things", "hilbig2014generalized",
    "jansen2021dunningkruger", "kool2016when", "kool2017cost",
    "lefebvre2017behavioural", "levering2020revisiting", "ludwig2023human",
    "peterson2021using", "plonsky2018when", "ruggeri2022globalizability",
    "sadeghiyeh2020temporal", "schulz2020finding", "somerville2017charting",
    "speekenbrink2008learning", "steingroever2015data", "tomov2021multitask",
    "waltz2020differential", "wilson2014humans", "wu2018generalisation",
    "wu2023chunking", "wulff2018description", "wulff2018sampling",
    "xiong2023neural", "zorowitz2023data",
)

# HF repo names that differ from the test-split family names
FAMILY_RENAMES = {"collsioo2023is": "collsiöö2023MCPL"}

# tasks whose canonical response lives outside the standard column
COLUMN_OVERRIDES = {"zorowitz2023data": ["choice_S1", "choice_S2"]}

def download_tables(cache):
    """Fetch any missing parquet tables into `cache`, one directory per task."""

    cache = Path(cache)
    for family in FAMILIES:
        target = cache / family
        if target.exists() and any(target.rglob("*.parquet")):
            continue
        files = fetch_json(f"{API}/{family}/tree/main?recursive=true")
        parquets = [f['path'] for f in files
                    if f['path'].endswith(".parquet")]
        print(f"downloading {family}: {len(parquets)} file(s)")
        for path in parquets:
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"{RESOLVE}/{family}/resolve/main/{path}", destination)

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)
