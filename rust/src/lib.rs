//! Rust extension module for swissphenocam.
//!
//! Exposes compute-heavy phenology kernels to Python as `swissphenocam._native`.
//! Pure logic lives in `core`; this file is the thin PyO3 binding layer.

pub mod core;
pub mod extraction;

use crate::extraction::compute_chromcoord_core;
use core::{
    calculate_transitions_pyhelper, cv_moving_average_core, cv_moving_average_from_perm,
    rolling_mean_centered,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Centered rolling mean with `min_periods=1`, ignoring NaNs.
///
/// Mirrors `pandas Series.rolling(window, center=True, min_periods=1).mean()`.
#[pyfunction]
fn rolling_mean_centered_rs(
    arr: PyReadonlyArray1<'_, f64>,
    window: usize,
) -> PyResult<Py<PyArray1<f64>>> {
    if window == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be >= 1",
        ));
    }
    if window % 2 == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "window must be odd to match centered definition",
        ));
    }
    let py = arr.py();
    let data = arr.as_slice()?;
    let out = PyArray1::from_vec(py, rolling_mean_centered(data, window));
    Ok(out.unbind())
}

/// Select best centered moving-average window via K-fold CV.
///
/// Mirrors `cv_moving_average_python` with `shuffle=True, random_state=42`.
/// Falls back to an internal Xoshiro RNG if the numpy RandomState call fails.
#[pyfunction]
fn cv_moving_average_rs(
    arr: PyReadonlyArray1<'_, f64>,
    window_candidates: Vec<usize>,
    num_splits: usize,
) -> PyResult<Option<usize>> {
    if num_splits == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_splits must be >= 1",
        ));
    }
    let data = arr.as_slice()?;
    let n = data.len();
    let py = arr.py();

    // Mirror sklearn KFold(shuffle=True, random_state=42) using
    // numpy.random.RandomState(42).permutation(n).
    let perm: Option<Vec<usize>> = (|| -> PyResult<Vec<usize>> {
        let np = py.import("numpy")?;
        let rs = np
            .getattr("random")?
            .call_method1("RandomState", (42u64,))?;
        rs.call_method1("permutation", (n,))?.extract::<Vec<usize>>()
    })()
    .ok()
    .filter(|p| p.len() == n);

    if let Some(perm) = perm {
        return Ok(cv_moving_average_from_perm(
            data,
            &window_candidates,
            num_splits,
            &perm,
        ));
    }

    Ok(cv_moving_average_core(
        data,
        &window_candidates,
        num_splits,
        42,
    ))
}

/// Extract phenological transition dates from a smoothed GCC series.
#[pyfunction]
fn calculate_transitions_rs(
    py: Python<'_>,
    doy_dates: PyReadonlyArray1<'_, i32>,
    gcc_values: PyReadonlyArray1<'_, f64>,
    thresholds: Vec<f64>,
    season_start: i32,
    season_end: i32,
    prefix: String,
) -> PyResult<Option<Py<PyAny>>> {
    let dates = doy_dates.as_slice()?;
    let values = gcc_values.as_slice()?;
    if dates.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "doy_dates and gcc_values must have the same length",
        ));
    }

    let Some(core) =
        calculate_transitions_pyhelper(dates, values, &thresholds, season_start, season_end, &prefix)
    else {
        return Ok(None);
    };

    let dates_dict = pyo3::types::PyDict::new(py);
    dates_dict.set_item("PEAK", core.peak_date)?;
    for (k, v) in core.dates {
        match v {
            Some(x) => dates_dict.set_item(k, x)?,
            None => dates_dict.set_item(k, py.None())?,
        }
    }

    let out = pyo3::types::PyDict::new(py);
    out.set_item("dates", dates_dict)?;
    out.set_item("amplitude", core.amplitude)?;
    out.set_item("baseline", core.baseline)?;
    out.set_item("peak_value", core.peak_value)?;
    Ok(Some(out.into()))
}

/// Per-image GCC/RCC/BCC kernel over flat (CSR-style) polygon masks.
///
/// * `img` — `(H, W, 3)` uint8 BGR image (as produced by `cv2.imread`).
///   Must be C-contiguous.
/// * `mask_row_ptr` — length `n_masks + 1`, uint32 CSR row pointer.
/// * `mask_pixels` — flat pixel indices (`row*W + col`), uint32.
/// * `mask_counts` — per-mask pixel counts, uint32, length `n_masks`.
///
/// Returns `(gcc, rcc, bcc, mean_brightness)` as four float64 arrays of length
/// `n_masks`. Brightness-zero guard mirrors the Python path.
#[pyfunction]
fn compute_chromcoord_rs<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
    mask_row_ptr: PyReadonlyArray1<'py, u32>,
    mask_pixels: PyReadonlyArray1<'py, u32>,
    mask_counts: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
    let shape = img.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "img must have shape (H, W, 3)",
        ));
    }
    let img_view = img.as_array();
    let img_slice = img_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("img must be C-contiguous uint8 BGR")
    })?;
    let row_ptr = mask_row_ptr.as_slice()?;
    let pixels = mask_pixels.as_slice()?;
    let counts = mask_counts.as_slice()?;

    if row_ptr.len() != counts.len() + 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "mask_row_ptr length must equal mask_counts length + 1",
        ));
    }
    if let Some(&last) = row_ptr.last() {
        if last as usize != pixels.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "mask_row_ptr[-1] must equal mask_pixels length",
            ));
        }
    }

    let n_pixels_img = shape[0] * shape[1];
    let out = py.allow_threads(|| {
        // Cheap sanity check inside GIL-free region.
        if let Some(&max_p) = pixels.iter().max() {
            if (max_p as usize) >= n_pixels_img {
                return None;
            }
        }
        Some(compute_chromcoord_core(img_slice, row_ptr, pixels, counts))
    });

    let out = out.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("mask_pixels contains an out-of-range index")
    })?;

    Ok((
        PyArray1::from_vec(py, out.gcc).unbind(),
        PyArray1::from_vec(py, out.rcc).unbind(),
        PyArray1::from_vec(py, out.bcc).unbind(),
        PyArray1::from_vec(py, out.mean_brightness).unbind(),
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_centered_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cv_moving_average_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_transitions_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_chromcoord_rs, m)?)?;
    Ok(())
}
