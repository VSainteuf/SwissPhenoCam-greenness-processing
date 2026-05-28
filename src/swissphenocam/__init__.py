"""SwissPhenoCam greenness processing pipeline.

Three stages:
    1. extraction  - image sequences + polygon ROIs -> raw GCC/RCC CSVs
    2. timeseries  - raw values -> filtered & aggregated 1D/3D products
    3. transitions - smoothing + amplitude thresholding -> phenological dates
"""

__version__ = "0.0.1"
