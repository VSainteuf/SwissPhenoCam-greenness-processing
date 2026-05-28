//! Per-image GCC/RCC/BCC kernel over flat (CSR-style) polygon masks.
//!
//! Input image is BGR interleaved u8 (`cv2.imread` layout). Masks are encoded
//! as a CSR row pointer + pixel index array: row `m` references pixels
//! `mask_pixels[mask_row_ptr[m]..mask_row_ptr[m+1]]`, where each pixel index
//! is a flat `row*W + col` offset into the image.
//!
//! Returns per-mask GCC, RCC, BCC, and mean brightness. Brightness-zero guard
//! mirrors the Python path: zero mean_brightness is replaced by 1e-6 before
//! computing ratios.

pub struct ChromCoordOut {
    pub gcc: Vec<f64>,
    pub rcc: Vec<f64>,
    pub bcc: Vec<f64>,
    pub mean_brightness: Vec<f64>,
}

/// Compute per-mask GCC, RCC, BCC, mean_brightness on a BGR u8 image.
///
/// `img_bgr` is `3 * H * W` bytes, interleaved (B, G, R, B, G, R, ...).
/// Pixel indices in `mask_pixels` are flat `row * W + col` indices.
pub fn compute_chromcoord_core(
    img_bgr: &[u8],
    mask_row_ptr: &[u32],
    mask_pixels: &[u32],
    mask_counts: &[u32],
) -> ChromCoordOut {
    let n_masks = mask_counts.len();
    assert_eq!(mask_row_ptr.len(), n_masks + 1);

    let mut gcc = vec![0.0f64; n_masks];
    let mut rcc = vec![0.0f64; n_masks];
    let mut bcc = vec![0.0f64; n_masks];
    let mut mean_bright = vec![0.0f64; n_masks];

    let n_pixels = img_bgr.len() / 3;

    for m in 0..n_masks {
        let start = mask_row_ptr[m] as usize;
        let end = mask_row_ptr[m + 1] as usize;
        let count = mask_counts[m];

        // Accumulate channel sums as u64 (safe: at most 2^24 pixels * 255 < 2^32).
        let mut sum_b: u64 = 0;
        let mut sum_g: u64 = 0;
        let mut sum_r: u64 = 0;

        for &p in &mask_pixels[start..end] {
            let p = p as usize;
            debug_assert!(p < n_pixels, "pixel index out of range");
            let base = 3 * p;
            // BGR interleaved: cv2.imread layout.
            sum_b += img_bgr[base] as u64;
            sum_g += img_bgr[base + 1] as u64;
            sum_r += img_bgr[base + 2] as u64;
        }

        if count == 0 {
            // Degenerate mask: mirror Python by emitting 0 / 1e-6 = 0.
            gcc[m] = 0.0;
            rcc[m] = 0.0;
            bcc[m] = 0.0;
            mean_bright[m] = 1e-6;
            continue;
        }

        let c = count as f64;
        let mean_g = sum_g as f64 / c;
        let mean_r = sum_r as f64 / c;
        let mean_b = sum_b as f64 / c;
        let mut mean_br = (sum_b + sum_g + sum_r) as f64 / c;

        // Guard: match Python `mean_brightness_per_mask[... == 0] = 1e-6`.
        if mean_br == 0.0 {
            mean_br = 1e-6;
        }

        gcc[m] = mean_g / mean_br;
        rcc[m] = mean_r / mean_br;
        bcc[m] = mean_b / mean_br;
        mean_bright[m] = mean_br;
    }

    // Suppress unused warning when debug_assert is stripped in release.
    let _ = n_pixels;

    ChromCoordOut { gcc, rcc, bcc, mean_brightness: mean_bright }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_mask_constant_image() {
        // 2x2 image, all pixels (10, 20, 30) BGR => brightness=60, GCC=20/60, RCC=30/60, BCC=10/60.
        let img = vec![10, 20, 30, 10, 20, 30, 10, 20, 30, 10, 20, 30];
        let row_ptr = vec![0u32, 4];
        let pixels = vec![0u32, 1, 2, 3];
        let counts = vec![4u32];

        let out = compute_chromcoord_core(&img, &row_ptr, &pixels, &counts);
        assert!((out.gcc[0] - 20.0 / 60.0).abs() < 1e-12);
        assert!((out.rcc[0] - 30.0 / 60.0).abs() < 1e-12);
        assert!((out.bcc[0] - 10.0 / 60.0).abs() < 1e-12);
        assert!((out.mean_brightness[0] - 60.0).abs() < 1e-12);
    }

    #[test]
    fn all_zero_image_brightness_guard() {
        let img = vec![0u8; 12];
        let row_ptr = vec![0u32, 4];
        let pixels = vec![0u32, 1, 2, 3];
        let counts = vec![4u32];
        let out = compute_chromcoord_core(&img, &row_ptr, &pixels, &counts);
        assert_eq!(out.mean_brightness[0], 1e-6);
        assert_eq!(out.gcc[0], 0.0);
        assert_eq!(out.rcc[0], 0.0);
        assert_eq!(out.bcc[0], 0.0);
    }

    #[test]
    fn two_masks_disjoint() {
        // 2x2: pixel 0 = (0,0,0); pixel 1 = (10,20,30); pixel 2 = (10,20,30); pixel 3 = (0,0,0)
        let img = vec![0, 0, 0, 10, 20, 30, 10, 20, 30, 0, 0, 0];
        let row_ptr = vec![0u32, 2, 4];
        let pixels = vec![1u32, 2, 0, 3];
        let counts = vec![2u32, 2];
        let out = compute_chromcoord_core(&img, &row_ptr, &pixels, &counts);
        assert!((out.gcc[0] - 20.0 / 60.0).abs() < 1e-12);
        assert!((out.rcc[0] - 30.0 / 60.0).abs() < 1e-12);
        assert!((out.bcc[0] - 10.0 / 60.0).abs() < 1e-12);
        assert_eq!(out.mean_brightness[1], 1e-6);
    }
}
