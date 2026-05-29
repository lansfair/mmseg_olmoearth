from __future__ import annotations

from dataclasses import dataclass


SENTINEL2_L2A_BANDS = [
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B8A",
    "B11",
    "B12",
    "B01",
    "B09",
]

SENTINEL1_BANDS = ["vv", "vh"]

RGB_TO_SENTINEL2_L2A = {"R": "B04", "G": "B03", "B": "B02"}


POTSDAM_CLASSES = (
    "impervious_surface",
    "building",
    "low_vegetation",
    "tree",
    "car",
    "clutter",
)
POTSDAM_PALETTE = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 255],
    [0, 255, 0],
    [255, 255, 0],
    [255, 0, 0],
]

CROP_TYPE_CLASSES = (
    "Background",
    "Lucerne/Medics",
    "Planted pastures (perennial)",
    "Fallow",
    "Wine grapes",
    "Weeds",
    "Small grain grazing",
    "Wheat",
    "Canola",
    "Rooibos",
)
CROP_TYPE_PALETTE = [
    [0, 0, 0],
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
]


PASTIS_CLASSES = (
    "background",
    "meadow",
    "soft_winter_wheat",
    "corn",
    "winter_barley",
    "winter_rapeseed",
    "spring_barley",
    "sunflower",
    "grapevine",
    "beet",
    "winter_triticale",
    "winter_durum_wheat",
    "fruits_vegetables_flowers",
    "potatoes",
    "leguminous_fodder",
    "soybeans",
    "orchard",
    "mixed_cereal",
    "sorghum",
)
PASTIS_PALETTE = [
    [166, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [255, 255, 153],
    [177, 89, 40],
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180, 98],
    [179, 222, 105],
]

SEN1FLOODS11_CLASSES = ("non_flood", "flood")
SEN1FLOODS11_PALETTE = [[0, 0, 0], [0, 120, 255]]

MADOS_CLASSES = (
    "marine_debris",
    "dense_sargassum",
    "sparse_floating_algae",
    "natural_organic_material",
    "ship",
    "oil_spill",
    "marine_water",
    "sediment_laden_water",
    "foam",
    "turbid_water",
    "shallow_water",
    "waves_and_wakes",
    "oil_platform",
    "jellyfish",
    "sea_snot",
)
MADOS_PALETTE = [
    [0, 0, 0],
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
]

AWF_CLASSES = (
    "woodland_forest",
    "open_water",
    "shrubland_savanna",
    "herbaceous_wetland",
    "grassland_barren",
    "agriculture_settlement",
    "montane_forest",
    "lava_forest",
    "urban_dense_development",
    "nodata",
)

AWF_PALETTE = [
    [38, 115, 77],
    [70, 130, 180],
    [143, 188, 143],
    [46, 139, 87],
    [189, 183, 107],
    [102, 205, 170],
    [34, 139, 34],
    [255, 140, 0],
    [178, 34, 34],
    [0, 0, 0],
]

NANDI_CLASSES = (
    "coffee",
    "grassland",
    "trees",
    "maize",
    "sugarcane",
    "tea",
    "vegetables",
    "legumes",
    "water",
    "builtup",
    "nodata",
)

NANDI_PALETTE = [
    [124, 67, 18],
    [169, 214, 146],
    [51, 160, 44],
    [240, 159, 28],
    [251, 154, 153],
    [31, 234, 146],
    [166, 206, 227],
    [227, 26, 28],
    [31, 120, 180],
    [136, 33, 233],
    [0, 0, 0],
]


@dataclass(frozen=True)
class ModalitySpec:
    name: str
    bands: tuple[str, ...]
    sample_field: str


MODALITY_SPECS = {
    "sentinel2_l2a": ModalitySpec(
        name="sentinel2_l2a",
        bands=tuple(SENTINEL2_L2A_BANDS),
        sample_field="sentinel2_l2a",
    ),
    "sentinel1": ModalitySpec(
        name="sentinel1",
        bands=tuple(SENTINEL1_BANDS),
        sample_field="sentinel1",
    ),
}


def get_modality_bands(modality: str) -> tuple[str, ...]:
    return MODALITY_SPECS[modality].bands


def get_sample_field(modality: str) -> str:
    return MODALITY_SPECS[modality].sample_field
