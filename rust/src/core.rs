use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn rolling_mean_centered_into(values: &[f64], window: usize, out: &mut [f64]) {
    assert!(
        window > 0 && window % 2 == 1,
        "window must be odd and >= 1 (got {window})"
    );
    let n = values.len();
    if n == 0 || out.len() != n {
        return;
    }
    let half = window / 2;

    let mut start = 0usize;
    let mut end = half.min(n - 1);
    let mut sum = 0.0f64;
    let mut cnt = 0u32;
    for &v in &values[start..=end] {
        if !v.is_nan() {
            sum += v;
            cnt += 1;
        }
    }

    for i in 0..n {
        let start_new = i.saturating_sub(half);
        let end_new = (i + half).min(n - 1);

        while start < start_new {
            let v = values[start];
            if !v.is_nan() {
                sum -= v;
                cnt -= 1;
            }
            start += 1;
        }

        while end < end_new {
            end += 1;
            let v = values[end];
            if !v.is_nan() {
                sum += v;
                cnt += 1;
            }
        }

        if cnt == 0 {
            out[i] = f64::NAN;
        } else {
            out[i] = sum / (cnt as f64);
        }
    }
}

pub fn interpolate_nans(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    interpolate_nans_in_place(&mut out);
    out
}

pub fn interpolate_nans_in_place(out: &mut [f64]) {
    let n = out.len();
    if n == 0 {
        return;
    }

    // Find first non-NaN
    let mut first_valid: Option<usize> = None;
    for i in 0..n {
        if !out[i].is_nan() {
            first_valid = Some(i);
            break;
        }
    }
    let Some(first_i) = first_valid else {
        return;
    };

    // Fill leading NaNs
    let first_val = out[first_i];
    for v in &mut out[..first_i] {
        *v = first_val;
    }

    // Fill interior gaps
    let mut last_valid = first_i;
    for i in (first_i + 1)..n {
        if !out[i].is_nan() {
            if i > last_valid + 1 {
                let left = out[last_valid];
                let right = out[i];
                let gap = i - last_valid;
                for k in 1..gap {
                    let frac = (k as f64) / (gap as f64);
                    out[last_valid + k] = left + (right - left) * frac;
                }
            }
            last_valid = i;
        }
    }

    // Fill trailing NaNs
    let last_val = out[last_valid];
    for v in &mut out[(last_valid + 1)..] {
        *v = last_val;
    }
}

/// Returns 0 for 0.0 input (matching `np.sign`), unlike Rust's `f64::signum` which returns +0.0/-0.0.
fn signum_like_numpy(x: f64) -> i8 {
    if x > 0.0 {
        1
    } else if x < 0.0 {
        -1
    } else {
        0
    }
}

fn find_crossing_date(
    dates: &[i32],
    values: &[f64],
    threshold: f64,
    direction_up: bool,
) -> Option<f64> {
    // Semantics match the Python reference in thresholds.py:
    //   up   -> last up-crossing   (latest confirmed green-up start)
    //   down -> first down-crossing (earliest green-down start)
    if dates.len() < 2 || values.len() < 2 {
        return None;
    }
    let mut prev_sign = signum_like_numpy(values[0] - threshold);
    let mut first_idx: Option<usize> = None;
    let mut last_idx: Option<usize> = None;
    for i in 0..(values.len() - 1) {
        let next_sign = signum_like_numpy(values[i + 1] - threshold);
        let diff_sign = (next_sign as i16) - (prev_sign as i16);
        if direction_up {
            if diff_sign > 0 {
                if first_idx.is_none() {
                    first_idx = Some(i);
                }
                last_idx = Some(i);
            }
        } else if diff_sign < 0 {
            if first_idx.is_none() {
                first_idx = Some(i);
            }
            last_idx = Some(i);
        }
        prev_sign = next_sign;
    }
    let idx = if direction_up { last_idx? } else { first_idx? };
    let x0 = dates[idx] as f64;
    let x1 = dates[idx + 1] as f64;
    let y0 = values[idx];
    let y1 = values[idx + 1];
    let denom = y1 - y0;
    if denom == 0.0 {
        return Some(x0);
    }
    let frac = (threshold - y0) / denom;
    Some(x0 + (x1 - x0) * frac)
}

#[derive(Debug, Clone, Copy)]
pub struct TransitionsMeta {
    pub peak_date: i32,
    pub peak_value: f64,
    pub baseline: f64,
    pub amplitude: f64,
}

/// Interpolate NaNs, restrict to DOY season, find peak/baseline, then compute
/// threshold crossings on the rising and falling halves of the curve.
pub fn calculate_transitions_core(
    doy_dates: &[i32],
    gcc_values: &[f64],
    thresholds_sorted: &[f64],
    season_start: i32,
    season_end: i32,
) -> Option<(TransitionsMeta, Vec<Option<f64>>)> {
    if doy_dates.len() != gcc_values.len() {
        return None;
    }
    if thresholds_sorted.is_empty() {
        return None;
    }

    let mut out_dates: Vec<Option<f64>> = vec![None; 2 * thresholds_sorted.len()];

    let mut interp = gcc_values.to_vec();
    interpolate_nans_in_place(&mut interp);

    let mut dates_f: Vec<i32> = Vec::with_capacity(doy_dates.len().min(512));
    let mut values_f: Vec<f64> = Vec::with_capacity(gcc_values.len().min(512));
    for (&d, &v) in doy_dates.iter().zip(interp.iter()) {
        if d >= season_start && d <= season_end {
            dates_f.push(d);
            values_f.push(v);
        }
    }
    if dates_f.is_empty() {
        return None;
    }
    if values_f.iter().all(|v| v.is_nan()) {
        return None;
    }

    // peak + baseline
    let mut peak_value = f64::NEG_INFINITY;
    let mut peak_idx = 0usize;
    let mut min_value = f64::INFINITY;
    for (i, &v) in values_f.iter().enumerate() {
        if v > peak_value {
            peak_value = v;
            peak_idx = i;
        }
        if v < min_value {
            min_value = v;
        }
    }
    let peak_date = dates_f[peak_idx];
    let amplitude = peak_value - min_value;

    let rising_dates = &dates_f[..=peak_idx];
    let rising_values = &values_f[..=peak_idx];
    let falling_dates = &dates_f[peak_idx..];
    let falling_values = &values_f[peak_idx..];

    for (i, t) in thresholds_sorted.iter().copied().enumerate() {
        let thresh_val = min_value + t * amplitude;
        out_dates[2 * i] = find_crossing_date(rising_dates, rising_values, thresh_val, true);
        out_dates[2 * i + 1] = find_crossing_date(falling_dates, falling_values, thresh_val, false);
    }

    Some((
        TransitionsMeta {
            peak_date,
            peak_value,
            baseline: min_value,
            amplitude,
        },
        out_dates,
    ))
}

/// NaN-skipping std-dev via Welford's online algorithm with Bessel correction (ddof=1), matching pandas default.
pub fn std_skip_nan(values: &[f64]) -> f64 {
    let mut count = 0usize;
    let mut mean = 0.0f64;
    let mut m2 = 0.0f64;
    for &x in values {
        if x.is_nan() {
            continue;
        }
        count += 1;
        let delta = x - mean;
        mean += delta / (count as f64);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    if count <= 1 {
        return f64::NAN;
    }
    (m2 / ((count - 1) as f64)).sqrt()
}

fn cv_fold_ranges(n: usize, num_splits: usize) -> Vec<(usize, usize)> {
    let base = n / num_splits;
    let remainder = n % num_splits;
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(num_splits);
    let mut start = 0usize;
    for i in 0..num_splits {
        let sz = if i < remainder { base + 1 } else { base };
        let end = start + sz;
        ranges.push((start, end));
        start = end;
    }
    ranges
}

fn build_fold_id(
    n: usize,
    fold_ranges: &[(usize, usize)],
    perm_indices: &[usize],
) -> Option<Vec<u16>> {
    let mut fold_id: Vec<u16> = vec![u16::MAX; n];
    for (fold_i, (start, end)) in fold_ranges.iter().copied().enumerate() {
        let tag = fold_i as u16;
        if end > perm_indices.len() {
            return None;
        }
        for &idx in &perm_indices[start..end] {
            if idx >= n {
                return None;
            }
            fold_id[idx] = tag;
        }
    }
    if fold_id.iter().any(|x| *x == u16::MAX) {
        return None;
    }
    Some(fold_id)
}

/// Leave-fold-out CV smoothing using prefix sums for O(n*k) per window instead of O(n*w*k).
pub fn cv_moving_average_from_perm(
    signal: &[f64],
    window_candidates: &[usize],
    num_splits: usize,
    perm_indices: &[usize],
) -> Option<usize> {
    let n = signal.len();
    if num_splits == 0 || n < num_splits || perm_indices.len() != n {
        return None;
    }

    let fold_ranges = cv_fold_ranges(n, num_splits);
    let fold_id = build_fold_id(n, &fold_ranges, perm_indices)?;
    let k = fold_ranges.len();
    if k == 0 {
        return None;
    }

    // Build global and per-fold prefix sums over the signal (NaN-aware).
    let stride = n + 1;
    let mut pref_sum = vec![0.0f64; stride];
    let mut pref_cnt = vec![0u32; stride];
    let mut fold_pref_sum = vec![0.0f64; k * stride];
    let mut fold_pref_cnt = vec![0u32; k * stride];

    pref_sum[0] = 0.0;
    pref_cnt[0] = 0;
    for (i, &v) in signal.iter().enumerate() {
        pref_sum[i + 1] = pref_sum[i];
        pref_cnt[i + 1] = pref_cnt[i];
        if !v.is_nan() {
            pref_sum[i + 1] += v;
            pref_cnt[i + 1] += 1;
        }
    }

    for fold_i in 0..k {
        let tag = fold_i as u16;
        let base = fold_i * stride;
        fold_pref_sum[base] = 0.0;
        fold_pref_cnt[base] = 0;
        for i in 0..n {
            let mut s = fold_pref_sum[base + i];
            let mut c = fold_pref_cnt[base + i];
            if fold_id[i] == tag {
                let v = signal[i];
                if !v.is_nan() {
                    s += v;
                    c += 1;
                }
            }
            fold_pref_sum[base + i + 1] = s;
            fold_pref_cnt[base + i + 1] = c;
        }
    }

    let mut best_window: Option<usize> = None;
    let mut best_score = f64::INFINITY;
    for &w in window_candidates {
        if w == 0 {
            continue;
        }
        let half = w / 2;

        let mut sum_fold_mse = 0.0f64;
        let mut folds_used = 0usize;
        for (fold_i, (start, end)) in fold_ranges.iter().copied().enumerate() {
            let mut se_sum = 0.0;
            let mut m = 0usize;
            if end > perm_indices.len() {
                return None;
            }
            let base = fold_i * stride;
            for &idx in &perm_indices[start..end] {
                if idx >= n {
                    return None;
                }
                let y_true = signal[idx];
                if y_true.is_nan() {
                    continue;
                }

                let start_w = idx.saturating_sub(half);
                let end_w = (idx + half).min(n - 1);

                let total_sum = pref_sum[end_w + 1] - pref_sum[start_w];
                let total_cnt = pref_cnt[end_w + 1] - pref_cnt[start_w];
                let fold_sum = fold_pref_sum[base + end_w + 1] - fold_pref_sum[base + start_w];
                let fold_cnt = fold_pref_cnt[base + end_w + 1] - fold_pref_cnt[base + start_w];
                // Subtract the held-out fold's contribution to get the leave-fold-out mean.
                let denom = total_cnt.saturating_sub(fold_cnt);
                if denom == 0 {
                    continue;
                }
                let y_pred = (total_sum - fold_sum) / (denom as f64);
                if y_pred.is_nan() {
                    continue;
                }
                let diff = y_true - y_pred;
                se_sum += diff * diff;
                m += 1;
            }
            if m > 0 {
                sum_fold_mse += se_sum / (m as f64);
                folds_used += 1;
            }
        }
        if folds_used > 0 {
            let mean_err = sum_fold_mse / (folds_used as f64);
            if mean_err < best_score {
                best_score = mean_err;
                best_window = Some(w);
            }
        }
    }

    best_window
}

//
// Functions used by python binding. Not needed for full rust script.
//
pub fn rolling_mean_centered(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let mut out = vec![0.0f64; n];
    rolling_mean_centered_into(values, window, &mut out);
    out
}

pub fn cv_moving_average_core(
    signal: &[f64],
    window_candidates: &[usize],
    num_splits: usize,
    seed: u64,
) -> Option<usize> {
    let n = signal.len();
    if num_splits == 0 || n < num_splits {
        return None;
    }

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    indices.shuffle(&mut rng);
    cv_moving_average_from_perm(signal, window_candidates, num_splits, &indices)
}
// End of functions used by python binding only.

// Only used for mc_uncertainties_from_noises_rs and mc_uncertainties_from_noises_rs.
// Both of which are in lib.rs, only to run tests to check parity with Python.
#[derive(Debug, Clone)]
pub struct TransitionsCore {
    pub peak_date: i32,
    pub peak_value: f64,
    pub baseline: f64,
    pub amplitude: f64,
    pub dates: Vec<(String, Option<f64>)>, // stable order (up/down per threshold)
}

pub fn calculate_transitions_pyhelper(
    doy_dates: &[i32],
    gcc_values: &[f64],
    thresholds: &[f64],
    season_start: i32,
    season_end: i32,
    prefix: &str,
) -> Option<TransitionsCore> {
    if doy_dates.len() != gcc_values.len() {
        return None;
    }
    let mut sorted_thresholds = thresholds.to_vec();
    sorted_thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let (meta, out_dates) = calculate_transitions_core(
        doy_dates,
        gcc_values,
        &sorted_thresholds,
        season_start,
        season_end,
    )?;

    let mut dates_out: Vec<(String, Option<f64>)> = Vec::with_capacity(out_dates.len());
    for (i, t) in sorted_thresholds.iter().copied().enumerate() {
        let pct = (t * 100.0).round() as i32;
        dates_out.push((format!("{prefix}U_{pct}"), out_dates[2 * i]));
        dates_out.push((format!("{prefix}D_{pct}"), out_dates[2 * i + 1]));
    }

    Some(TransitionsCore {
        peak_date: meta.peak_date,
        peak_value: meta.peak_value,
        baseline: meta.baseline,
        amplitude: meta.amplitude,
        dates: dates_out,
    })
}
// End of functions used for testing only.
