# Spectrum Forming for Blade Tip Timing (BTT) Signal Processing

This repository provides the Python implementation for the numerical simulation presented in the paper: **"Spectrum forming and its high-performance implementations for the blade tip timing signal processing"**.

The code, `sf_mssp.py`, partially reproduces the results from the simulation analysis (Section 5) of the paper, demonstrating the effectiveness of the proposed Spectrum Forming (SF) framework and its high-performance variations.

## About the Paper

**Published in:** *Mechanical Systems and Signal Processing*, Volume 238, 2025, 113161.

**Authors:** Chenyu Zhang, Youhong Xiao, Zhicheng Xiao, Liang Yu, Jérôme Antoni.

### Abstract
> Blade Tip Timing (BTT) is a critical non-contact technique for monitoring rotating blade vibrations, yet its effectiveness is hindered by under-sampled signals that violate the Nyquist criterion. This paper introduces Spectrum Forming (SF), a novel framework tailored for BTT signal analysis, to address spectral aliasing and enhance vibration feature extraction. Building on SF, advanced methods—including non-negative least squares (De-NNLS), non-convex optimization with generalized mini-max concave penalty (De-GMCP), CLEAN based on frequency coherence (CLEAN-FC), and functional spectrum forming (FSF)—are developed to suppress aliasing and improve resolution. Numerical simulations and experimental studies on rotating blade disks and compressor rotors validate the efficacy of these methods.

## Code Description

The script `sf_mssp.py` simulates a blade vibration signal, samples it using a set of Time-of-Arrival (TOA) probes as described in the paper, and then processes the under-sampled signal to reconstruct its frequency spectrum.

The following algorithms from the paper are implemented:
* **Spectrum Forming (SF):** The baseline method for BTT spectral analysis (Section 3).
* **Deconvolution with NNLS (De-NNLS):** A deconvolution method based on non-negative least squares regression (Section 4.2.1).
* **Deconvolution with GMCP (De-GMCP):** A deconvolution method using a non-convex generalized mini-max concave penalty to induce sparsity (Section 4.2.2).
* **CLEAN based on Frequency Coherence (CLEAN-FC):** An iterative algorithm to remove aliased components from the spectrum (Section 4.2.3).
* **Functional Spectrum Forming (FSF):** A method that uses power properties to suppress spectral aliasing (Section 4.3).

## Requirements

The code is written in Python and requires the following libraries:
* NumPy
* SciPy
* Matplotlib

You can install these dependencies using pip:
```bash
pip install numpy scipy matplotlib
```

## Usage

To run the simulation, simply execute the Python script from your terminal:
```bash
python sf_mssp.py
```
The script will run the simulation with the parameters defined in the "Main" section. Upon completion, it will display a plot containing five subplots. Each subplot shows the reconstructed single-sided amplitude spectrum for one of the implemented methods, allowing for a direct comparison of their performance in resolving the true signal frequencies (45 Hz and 220 Hz) and suppressing aliasing.

## Citation

If you use this code or the methods from the paper in your research, please cite the original publication:

```bibtex
@article{Zhang2025SF,
  title   = {Spectrum forming and its high-performance implementations for the blade tip timing signal processing},
  author  = {Chenyu Zhang and Youhong Xiao and Zhicheng Xiao and Liang Yu and J{\'e}r{\^o}me Antoni},
  journal = {Mechanical Systems and Signal Processing},
  volume  = {238},
  pages   = {113161},
  year    = {2025},
  doi     = {10.1016/j.ymssp.2025.113161},
  issn    = {0888-3270}
}
```

## Author

* **Chenyu Zhang**

## License

This project is open-sourced under the MIT License. Please see the `LICENSE` file for more details.
