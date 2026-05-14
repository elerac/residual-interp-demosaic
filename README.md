# Residual Interpolation Demosaicing

Python re-implementation of residual interpolation demosaicing algorithms from the Institute of Science Tokyo (formerly Tokyo Institute of Technology) project page, [Residual Interpolation for Color Image Demosaicking](http://www.ok.sc.e.titech.ac.jp/res/DM/RI.html).

This repository provides a Python re-implementation only. It is not the original MATLAB release and is not an official distribution from the original authors. The upstream MATLAB code and publications were released by Yusuke Monno and Daisuke Kiku.

The public `demosaic()` helper uses OpenCV-style code names and works with normal full-color OpenCV BGR images by generating a Bayer mosaic internally, then reconstructing the color image.

## Usage

```python
import cv2

from demosaic import demosaic

image = cv2.imread("tshirts.jpg", cv2.IMREAD_COLOR)
result = demosaic(image, "COLOR_BayerRGGB2BGR_ARI")
cv2.imwrite("tshirts_ari.png", result.astype("uint8"))
```

Codes use this form:

```python
COLOR_Bayer{PATTERN}2BGR_{ALGORITHM}
```

### Supported Algorithms

| Method | Corresponding paper |
| --- | --- |
| `RI` | Kiku et al., **"Residual Interpolation for Color Image Demosaicking,"** IEEE ICIP 2013. |
| `MLRI` | Kiku et al., **"Minimized-Laplacian Residual Interpolation for Color Image Demosaicking,"** IS&T/SPIE Electronic Imaging 2014. |
| `MLRI2` | Kiku et al., **"Beyond Color Difference: Residual Interpolation for Color Image Demosaicking,"** IEEE Transactions on Image Processing 2016. |
| `ARI` | Monno et al., **"Adaptive Residual Interpolation for Color Image Demosaicking,"** IEEE ICIP 2015. |
| `ARI2` | Monno et al., **"Adaptive Residual Interpolation for Color and Multispectral Image Demosaicking,"** Sensors 2017. |

### Supported Bayer Patterns

| Pattern | CFA Layout |
| --- | --- |
| `RGGB` | R G<br>G B |
| `GRBG` | G R<br>B G |
| `GBRG` | G B<br>R G |
| `BGGR` | B G<br>G R |

## Benchmark

The specialized benchmark compares all five algorithms in this repository with OpenCV bilinear/edge-aware demosaicing and `colour_demosaicing` Malvar2004/Menon2007. It creates one shared Bayer CFA from `tshirts.jpg`, runs every method against that same CFA.

```bash
python benchmark_tshirts.py --runs 5
```

Benchmark image: `tshirts.jpg` (1500 x 1000), Bayer pattern `RGGB`, 5 timed runs after one warmup. CPSNR is computed against the original RGB image with `peak=255` and no border crop. SSIM Avg is the RGB-channel average from `demosaic.ssim`.

| Method | Implementation | CPSNR (dB) | SSIM Avg | Time (s) |
| --- | --- | ---: | ---: | ---: |
| ARI2 | This repository | *38.86* | **0.9902** | 103.6532 |
| ARI | This repository | **38.88** | *0.9898* | 87.4439 |
| MLRI2 | This repository | 38.29 | 0.9892 | 11.9840 |
| MLRI | This repository | 38.04 | 0.9887 | 7.7656 |
| RI | This repository | 38.03 | 0.9885 | 3.7507 |
| Menon2007 | colour_demosaicing | 35.69 | 0.9817 | 0.2246 |
| Malvar2004 | colour_demosaicing | 34.54 | 0.9772 | 0.0770 |
| Edge-Aware | OpenCV | 30.77 | 0.9511 | 0.0002 |
| Bilinear | OpenCV | 30.57 | 0.9500 | 0.0002 |

## Input and CFA

| Input | Bayer CFA (RGB-colored) |
| --- | --- |
| ![input](results/tshirts/tshirts_input.png) | ![cfa](results/tshirts/tshirts_cfa_rgb.png) |

## CFA Crops

4x nearest-neighbor zoom.

| Crop Region | Original Input | RGB-colored CFA |
| --- | --- | --- |
| crop1 | ![input crop1](results/tshirts/tshirts_input_crop1.png) | ![cfa crop1](results/tshirts/tshirts_cfa_rgb_crop1.png) |
| crop2 | ![input crop2](results/tshirts/tshirts_input_crop2.png) | ![cfa crop2](results/tshirts/tshirts_cfa_rgb_crop2.png) |

## Demosaiced Crops

4x nearest-neighbor zoom.

| Method | Crop1 | Crop2 |
| :---: | :---: | :---: |
| Original | ![original crop1](results/tshirts/tshirts_input_crop1.png) | ![original crop2](results/tshirts/tshirts_input_crop2.png) |
| RI | ![ri crop1](results/tshirts/tshirts_demosaiced_ri_crop1.png) | ![ri crop2](results/tshirts/tshirts_demosaiced_ri_crop2.png) |
| MLRI | ![mlri crop1](results/tshirts/tshirts_demosaiced_mlri_crop1.png) | ![mlri crop2](results/tshirts/tshirts_demosaiced_mlri_crop2.png) |
| MLRI2 | ![mlri2 crop1](results/tshirts/tshirts_demosaiced_mlri2_crop1.png) | ![mlri2 crop2](results/tshirts/tshirts_demosaiced_mlri2_crop2.png) |
| ARI | ![ari crop1](results/tshirts/tshirts_demosaiced_ari_crop1.png) | ![ari crop2](results/tshirts/tshirts_demosaiced_ari_crop2.png) |
| ARI2 | ![ari2 crop1](results/tshirts/tshirts_demosaiced_ari2_crop1.png) | ![ari2 crop2](results/tshirts/tshirts_demosaiced_ari2_crop2.png) |
| Bilinear | ![opencv_bilinear crop1](results/tshirts/tshirts_demosaiced_opencv_bilinear_crop1.png) | ![opencv_bilinear crop2](results/tshirts/tshirts_demosaiced_opencv_bilinear_crop2.png) |
| Edge-Aware | ![opencv_ea crop1](results/tshirts/tshirts_demosaiced_opencv_ea_crop1.png) | ![opencv_ea crop2](results/tshirts/tshirts_demosaiced_opencv_ea_crop2.png) |
| Malvar2004 | ![colour_malvar2004 crop1](results/tshirts/tshirts_demosaiced_colour_malvar2004_crop1.png) | ![colour_malvar2004 crop2](results/tshirts/tshirts_demosaiced_colour_malvar2004_crop2.png) |
| Menon2007 | ![colour_menon2007 crop1](results/tshirts/tshirts_demosaiced_colour_menon2007_crop1.png) | ![colour_menon2007 crop2](results/tshirts/tshirts_demosaiced_colour_menon2007_crop2.png) |
