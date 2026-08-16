"""Scenario prompts, sourced from `scenario_molmo_prompt.txt` (single source of truth
the user edits) instead of being hard-coded. The file lists the Molmo prompts per
label/scenario; we extract the quoted string for each.

  Molmo(place)  table -> parent place point (table centre)
  Molmo(place)  fruit -> NOT USED (fruit parent uses the SAM3 text prompt below)
  Molmo(grasp)        -> child grasp point (FIXED prompt, object-agnostic)
  Molmo(local_place)  -> fruit tray empty holes (multi)

The fruit parent SAM3 text/concept prompt is a noun phrase (spec step 3-5-A) — the
tray object name, kept here next to the Molmo prompts.
"""
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPT_FILE = os.path.join(REPO, "scenario_molmo_prompt.txt")

# SAM3 concept/text prompt(s) for the fruit-tray parent segmentation (spec 3-5-A). SAM3 text
# grounding is prompt-sensitive PER SCENE: one phrase segments one tray but returns an EMPTY mask
# on another (verified — the old 'molded fiber fruit tray' grounds one tray at 89k px but 0 px on
# a different tray, where 'fruit tray'/'egg carton'/'white tray' all work). So we TRY THESE IN
# ORDER until one yields a non-empty tray cloud. 'fruit tray'/'egg carton'/'white tray' segmented
# BOTH test trays; the rest are extra fallbacks.
SAM_FRUIT_TRAY_PROMPTS = ["fruit tray", "egg carton", "white tray", "tray",
                          "molded fiber fruit tray", "molded pulp tray"]
SAM_FRUIT_TRAY = SAM_FRUIT_TRAY_PROMPTS[0]                    # primary (back-compat)

_CACHE = None


def _load():
    prompts = {}
    with open(PROMPT_FILE, encoding="utf-8") as f:
        for line in f:
            if "Molmo(place)" not in line and "Molmo(grasp)" not in line \
                    and "Molmo(local_place)" not in line:
                continue
            q = re.search(r'"([^"]+)"', line)
            if not q:
                continue
            val = q.group(1)
            if "Molmo(place)" in line and "table" in line:
                prompts["place_table"] = val
            elif "Molmo(place)" in line and "fruit" in line:
                prompts["place_fruit"] = val
            elif "Molmo(grasp)" in line:
                prompts["grasp"] = val
            elif "Molmo(local_place)" in line:
                prompts["local_place"] = val
    return prompts


def _p():
    global _CACHE
    if _CACHE is None:
        _CACHE = _load()
    return _CACHE


def place_prompt(scenario):
    """Molmo(place) prompt. Only used for `table` (fruit parent uses SAM3 text)."""
    return _p()["place_table" if scenario == "table" else "place_fruit"]


def grasp_prompt():
    """Molmo(grasp) prompt — FIXED ("Point to the object being held by the robot hand")."""
    return _p()["grasp"]


def local_place_prompt():
    return _p()["local_place"]
