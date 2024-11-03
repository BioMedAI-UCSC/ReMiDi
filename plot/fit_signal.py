import numpy as np
from plot.process_signal_poly import process_signal_poly


def fit_signal(signal, signal_allcmpts, bvalues):
    """
    Fit the ADC from the dMRI signal.

    Parameters:
        signal (array): Signal intensities for different compartments.
                        Shape: (ncompartment, namplitude, nsequence, ndirection)
        signal_allcmpts (array): Combined signal intensities from all compartments.
                                 Shape: (namplitude, nsequence, ndirection)
        bvalues (array): b-values.
                         Shape: (namplitude, nsequence)

    Returns:
        results (dict): A dictionary containing the fitted ADC values for each compartment
                        and combined, along with S0 values for each compartment and combined.
    """
    ncompartment, namplitude, nsequence, ndirection = signal.shape

    if namplitude == 1:
        raise ValueError("Cannot fit ADC from one b-value only.")

    adc = np.zeros((ncompartment, nsequence, ndirection))
    adc_allcmpts = np.zeros((nsequence, ndirection))
    S0 = np.zeros((ncompartment, nsequence, ndirection))
    S0_allcmpts = np.zeros((nsequence, ndirection))

    for idir in range(ndirection):
        for iseq in range(nsequence):
            b = bvalues[:, iseq]
            bmin = b[0]
            bmax = b[-1]
            for icmpt in range(ncompartment):
                data = np.real(signal[icmpt, :, iseq, idir])
                adc_fit, _, S01d = process_signal_poly(data, b, bmin, bmax)
                adc[icmpt, iseq, idir] = adc_fit
                S0[icmpt, iseq, idir] = S01d
            data_allcmpts = np.real(signal_allcmpts[:, iseq, idir])
            adc_fit_allcmpts, _, S01d_allcmpts = process_signal_poly(
                data_allcmpts, b, bmin, bmax
            )
            adc_allcmpts[iseq, idir] = adc_fit_allcmpts
            S0_allcmpts[iseq, idir] = S01d_allcmpts

    results = {
        "adc": adc,
        "adc_allcmpts": adc_allcmpts,
        "S0": S0,
        "S0_allcmpts": S0_allcmpts,
    }

    return results
