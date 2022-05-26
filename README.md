# digital-signal-processing

## Table of Contents

- [Table of Contents](#table-of-contents)
- [中文报告](#中文报告)
- [Getting Started](#getting-started)
  - [0 Prerequisites](#0-prerequisites)
  - [1 Installation](#1-installation)
  - [2 Usage](#2-usage)
- [Contributors](#contributors)
- [License](#license)

## 中文报告

1. [FFT: Fast Fourier Transform](./docs/fft.md)
2. [语谱图](./docs/spectrogram.md)
3. [MFCC: Mel-Frequency Cepstral Coefficients](./docs/mfcc.md)

## Getting Started

### 0 Prerequisites

To set up the environment, you need to have the following dependencies installed.

- [Anaconda](https://www.anaconda.com/products/individual) 4.12 or later (with Python 3.9)

### 1 Installation

```bash
conda env update --name dsp --file environment.yml
conda activate dsp
```

### 2 Usage

Put the audio files in `./data`, and execute the following command.

```bash
python3 main.py
```

The generated figures are saved to `./assets`.

## Contributors

- [**Hakula Chen**](https://github.com/hakula139)<[i@hakula.xyz](mailto:i@hakula.xyz)> - Fudan University

## License

This repository is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
