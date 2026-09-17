"""
Transit detection with Box Least Squares.

BLS (Kovacs, Zucker & Mazeh 2002) fits a periodic box-shaped dip: the model is
a step function parameterised by period, epoch, duration and depth. It is the
standard transit-search algorithm because a transit genuinely is closer to a box
than to a sinusoid, and a Lomb-Scargle periodogram tuned for sinusoids responds
poorly to a signal that is flat for 99% of its phase.

Uses `astropy.timeseries.BoxLeastSquares`.

What a detection is and is not
------------------------------
A BLS peak is a PERIODIC DIP. Eclipsing binaries, background eclipsing binaries,
stellar pulsation, spot-crossing and instrumental artefacts all produce periodic
dips. The same false-positive taxonomy that motivates the tabular pipeline's
physics-consistency features applies here.

Two cheap sanity checks are therefore computed alongside every detection:

* `odd_even_depth_ratio` — alternating transit depths indicate a background
  eclipsing binary whose true period is twice the detected one.
* `secondary_eclipse_depth` — a dip at phase 0.5 indicates a self-luminous
  companion, i.e. a star.

Neither is conclusive. Both are reported rather than used to auto-reject.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from lightcurve.acquisition.load_lightcurve import LightCurveData

logger = logging.getLogger(__name__)

#: The Kepler pipeline's Multiple Event Statistic threshold. Below 7.1 sigma the
#: expected number of purely statistical false alarms over the mission exceeds one.
MES_THRESHOLD = 7.1


@dataclass
class TransitSearchResult:
    best_period_days: float
    best_epoch_bkjd: float
    best_duration_hours: float
    depth_ppm: float
    depth_snr: float
    power: float
    n_transits: int
    odd_even_depth_ratio: float | None = None
    secondary_eclipse_depth_ppm: float | None = None
    periodogram: dict[str, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    caveat: str = (
        "A BLS detection is a periodic dip, not a planet. Eclipsing binaries, "
        "background binaries, stellar variability and instrumental artefacts all "
        "produce periodic dips.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bls_search(
    lc: LightCurveData,
    min_period_days: float = 0.5,
    max_period_days: float | None = None,
    n_periods: int = 6000,
    durations_hours: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0),
    max_periodogram_points: int = 800,
) -> TransitSearchResult:
    """Run BLS on the flattened flux and characterise the strongest peak."""
    from astropy.timeseries import BoxLeastSquares

    flux = lc.flattened_flux if lc.flattened_flux is not None else lc.flux
    time = lc.time
    mask = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[mask], flux[mask]
    if len(time) < 50:
        raise ValueError("too few finite points for a transit search")

    baseline = float(time.max() - time.min())
    # Cap at baseline/3: detecting a period needs at least ~3 transits, so
    # anything longer is an artefact of the window, not a detection.
    if max_period_days is None:
        max_period_days = max(min_period_days * 2, baseline / 3.0)

    periods = np.linspace(min_period_days, max_period_days, n_periods)
    durations = np.asarray(durations_hours) / 24.0
    durations = durations[durations < min_period_days * 0.4]
    if durations.size == 0:
        durations = np.array([min_period_days * 0.05])

    bls = BoxLeastSquares(time, flux)
    power = bls.power(periods, durations)

    i = int(np.nanargmax(power.power))
    period = float(power.period[i])
    epoch = float(power.transit_time[i])
    duration = float(power.duration[i])
    depth = float(power.depth[i])

    # SNR is computed EMPIRICALLY rather than taken from power.depth_err.
    # BoxLeastSquares assumes unit per-point uncertainty when no `dy` is passed,
    # which makes its depth_err meaningless for normalised flux (it comes out
    # ~1/sqrt(N), i.e. thousands of times the real scatter). The honest estimate
    # is the depth divided by the standard error of the in-transit mean, using
    # an outlier-robust MAD scatter measured out of transit.
    snr = _empirical_snr(time, flux, period, epoch, duration, depth)

    n_transits = int(np.floor(baseline / period))
    warnings_: list[str] = []
    if np.isfinite(snr) and snr < MES_THRESHOLD:
        warnings_.append(
            f"Depth SNR {snr:.1f} is below the {MES_THRESHOLD} sigma threshold used by "
            "the Kepler pipeline; this is consistent with a noise fluctuation.")
    if n_transits < 3:
        warnings_.append(
            f"Only ~{n_transits} transits fall within the observed baseline. A period "
            "cannot be established reliably from fewer than three.")

    odd_even = _odd_even_ratio(time, flux, period, epoch, duration)
    if odd_even is not None and (odd_even < 0.7 or odd_even > 1.4):
        warnings_.append(
            f"Odd and even transit depths differ by a factor of {odd_even:.2f}. This is "
            "a classic signature of a background eclipsing binary at twice this period.")

    secondary = _secondary_depth(time, flux, period, epoch, duration)
    if secondary is not None and secondary > 0.25 * abs(depth) * 1e6:
        warnings_.append(
            "A significant dip is present near phase 0.5, consistent with a secondary "
            "eclipse from a self-luminous companion (i.e. a star, not a planet).")

    stride = max(1, len(power.period) // max_periodogram_points)
    return TransitSearchResult(
        best_period_days=round(period, 6),
        best_epoch_bkjd=round(epoch, 6),
        best_duration_hours=round(duration * 24, 4),
        depth_ppm=round(depth * 1e6, 2),
        depth_snr=round(snr, 3) if np.isfinite(snr) else float("nan"),
        power=round(float(power.power[i]), 6),
        n_transits=n_transits,
        odd_even_depth_ratio=round(odd_even, 4) if odd_even is not None else None,
        secondary_eclipse_depth_ppm=round(secondary, 2) if secondary is not None else None,
        periodogram={"period": [round(float(p), 5) for p in power.period[::stride]],
                     "power": [round(float(v), 6) for v in power.power[::stride]]},
        warnings=warnings_)


def _empirical_snr(time, flux, period, epoch, duration, depth) -> float:
    """depth / (robust out-of-transit scatter / sqrt(N_in_transit))."""
    phase = fold(time, period, epoch)
    half = (duration / period) / 2
    in_transit = np.abs(phase) < half
    out = np.abs(phase) > 3 * half
    n_in = int(in_transit.sum())
    if n_in < 3 or out.sum() < 20:
        return float("nan")
    resid = flux[out] - np.median(flux[out])
    scatter = 1.4826 * np.median(np.abs(resid))       # MAD -> sigma
    if not np.isfinite(scatter) or scatter <= 0:
        return float("nan")
    return float(abs(depth) / (scatter / np.sqrt(n_in)))


def fold(time: np.ndarray, period: float, epoch: float) -> np.ndarray:
    """Phase in [-0.5, 0.5), with 0 at mid-transit."""
    return ((time - epoch + 0.5 * period) % period) / period - 0.5


def _odd_even_ratio(time, flux, period, epoch, duration) -> float | None:
    """Mean depth of odd-numbered transits over even-numbered ones."""
    epoch_number = np.round((time - epoch) / period).astype(int)
    phase = fold(time, period, epoch)
    in_transit = np.abs(phase) < (duration / period) / 2
    out = np.abs(phase) > 0.25
    if in_transit.sum() < 8 or out.sum() < 20:
        return None
    base = float(np.median(flux[out]))
    odd = flux[in_transit & (epoch_number % 2 == 1)]
    even = flux[in_transit & (epoch_number % 2 == 0)]
    if len(odd) < 3 or len(even) < 3:
        return None
    d_odd, d_even = base - float(np.median(odd)), base - float(np.median(even))
    return float(d_odd / d_even) if d_even != 0 else None


def _secondary_depth(time, flux, period, epoch, duration) -> float | None:
    """Depth in ppm of any dip near phase 0.5."""
    phase = fold(time, period, epoch)
    half_width = (duration / period) / 2
    secondary = np.abs(np.abs(phase) - 0.5) < half_width
    out = np.abs(phase - 0.25) < 0.05
    if secondary.sum() < 5 or out.sum() < 10:
        return None
    return float((np.median(flux[out]) - np.median(flux[secondary])) * 1e6)
