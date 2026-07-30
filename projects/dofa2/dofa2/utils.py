"""Shared constants for the DOFAv2 MMSegmentation project."""

from __future__ import annotations

WAVELENGTHS = {
    'COASTAL_AEROSOL': 0.44,
    'BLUE': 0.49,
    'GREEN': 0.56,
    'RED': 0.665,
    'RED_EDGE_1': 0.705,
    'RED_EDGE_2': 0.74,
    'RED_EDGE_3': 0.783,
    'NIR_BROAD': 0.832,
    'NIR_NARROW': 0.864,
    'WATER_VAPOR': 0.945,
    'CIRRUS': 1.373,
    'SWIR_1': 1.61,
    'SWIR_2': 2.20,
    'THERMAL_INFRARED_1': 10.90,
    'THERMAL_INFRARED_2': 12.00,
    'VV': 5.405,
    'VH': 5.405,
    'ASC_VV': 5.405,
    'ASC_VH': 5.405,
    'DSC_VV': 5.405,
    'DSC_VH': 5.405,
    'VV-VH': 5.405,
}

ARCH_SETTINGS = {
    'base': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'default_out_indices': (2, 5, 8, 11),
    },
    'large': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'default_out_indices': (5, 11, 17, 23),
    },
}


def get_wavelengths(model_bands: tuple[str, ...] | list[str]) -> list[float]:
    """Return central wavelengths in micrometres for ``model_bands``."""
    wavelengths = []
    for band in model_bands:
        key = band.split('.')[-1].upper()
        if key not in WAVELENGTHS:
            supported = ', '.join(sorted(WAVELENGTHS))
            raise KeyError(f'Unknown DOFA band {band!r}. Supported bands: {supported}')
        wavelengths.append(WAVELENGTHS[key])
    return wavelengths


def get_arch_setting(arch: str) -> dict:
    """Return a copy of the requested architecture settings."""
    if arch not in ARCH_SETTINGS:
        raise KeyError(
            f'Unsupported DOFAv2 architecture {arch!r}; '
            f'choose one of {tuple(ARCH_SETTINGS)}.')
    return ARCH_SETTINGS[arch].copy()
