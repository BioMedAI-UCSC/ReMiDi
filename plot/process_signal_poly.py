import numpy as np


def process_signal_poly(data, bvalues, bmin, bmax):
    """
    Fit polynomial to log-signal against the b-values and calculate ADC, kurtosis, and S0.
    """
    # Filter data and bvalues based on bmin and bmax
    ind_b = (bvalues >= bmin) & (bvalues <= bmax)
    bvalues = bvalues[ind_b]
    data = data[ind_b]
    logdata = np.log(np.abs(data))

    namplitude = len(bvalues)

    if namplitude == 1:
        raise ValueError("Cannot fit data from one b-value only.")

    adc_old = 1
    kur_old = 1
    degree = 0
    found = False

    while not found and degree <= namplitude - 2:
        degree += 1
        coeffs = np.polyfit(bvalues, logdata, degree)
        adc = -coeffs[-2] if degree >= 1 else 0  # Correct indexing for adc

        if degree == 1:
            kur = 0
        elif degree >= 2:
            kur = 2 * coeffs[-3] / adc**2  # Correct indexing and calculation for kur

        diff = np.polyval(coeffs, bvalues) - logdata
        converged = np.max(np.abs(diff)) <= 1e-3 * np.max(np.abs(logdata))
        adc_stable = abs(adc - adc_old) <= 1e-6 or abs(adc - adc_old) <= 0.05 * abs(
            adc_old
        )
        kur_stable = (
            kur_old <= 0.15
            or abs(kur - kur_old) < 0.15 * abs(kur_old)
            or abs(kur - kur_old) < 0.15
        )

        if converged and adc_stable and kur_stable:
            found = True

        adc_old = adc
        kur_old = kur

    if not found:
        print("Warning: Kurtosis may not be accurate.")

    S0 = np.exp(coeffs[-1])

    return adc, kur, S0
