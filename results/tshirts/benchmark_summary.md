Benchmark image: `tshirts.jpg` (1500 x 1000), Bayer pattern `RGGB`, 5 timed runs after one warmup.

| Method | Implementation | CPSNR (dB) | SSIM Avg | Time (s) [mean +/- std] |
| --- | --- | ---: | ---: | ---: |
| ARI | This repository | 38.88 | 0.9898 | 87.4439+/-0.3120 |
| ARI2 | This repository | 38.86 | 0.9902 | 103.6532+/-0.3959 |
| MLRI2 | This repository | 38.29 | 0.9892 | 11.9840+/-0.0376 |
| MLRI | This repository | 38.04 | 0.9887 | 7.7656+/-0.0184 |
| RI | This repository | 38.03 | 0.9885 | 3.7507+/-0.0089 |
| Menon2007 | colour_demosaicing | 35.69 | 0.9817 | 0.2246+/-0.0034 |
| Malvar2004 | colour_demosaicing | 34.54 | 0.9772 | 0.0770+/-0.0012 |
| Edge-Aware | OpenCV | 30.77 | 0.9511 | 0.0002+/-0.0000 |
| Bilinear | OpenCV | 30.57 | 0.9500 | 0.0002+/-0.0001 |

### Input and CFA

| Input (BGR) | Bayer CFA (RGB-colored) |
| --- | --- |
| ![input](results/tshirts/tshirts_input.png) | ![cfa](results/tshirts/tshirts_cfa_rgb.png) |

### CFA Cropped Images (4x nearest-neighbor zoom)

| Crop Region | Original Input | RGB-colored CFA |
| --- | --- | --- |
| crop1 | ![input crop1](results/tshirts/tshirts_input_crop1.png) | ![cfa crop1](results/tshirts/tshirts_cfa_rgb_crop1.png) |
| crop2 | ![input crop2](results/tshirts/tshirts_input_crop2.png) | ![cfa crop2](results/tshirts/tshirts_cfa_rgb_crop2.png) |

### Demosaiced Cropped Images (4x nearest-neighbor zoom)

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
