
# Project 1:




## Part I: Histogram Computation

- Read the grayscale image `test.jpg`.
- Compute a 64-bin gray scale histogram without using built-in histogram functions.
- Plot the computed histogram.
- Compare with `numpy`'s histogram function by plotting both histograms side-by-side.

### Expected Output

- Two histograms that should be identical.

## Part II: Histogram Equalization

- Perform 64-bin grayscale histogram equalization on `test.jpg`.
- Plot the original image and histogram, alongside the equalized image and histogram.
- No skimage functions (or equivalents) are to be used for this part.

### Expected Output

- Original and equalized image with their respective histograms.

## Part III: Histogram Comparing

- Compare the 256-bin histograms of `day.jpg` and `night.jpg`.
- Convert images to grayscale and compute their histograms.
- Print the Bhattacharyya Coefficient of the two histograms.

### Expected Output

- Bhattacharyya Coefficient value (example: `0.8671201323799057`).

## Part IV: Histogram Matching

- Match the histograms of `day.jpg` and `night.jpg`.

### Grayscale Histogram Matching

- Match 256-bin histogram of the `day` image to the `night` image, creating a darker version of the `day` image.
- Display the grayscale `day`, `night`, and matched `day` images side by side.

### RGB Histogram Matching

- Repeat the grayscale matching process for each channel of the RGB images.
- Show the RGB `day`, `night`, and matched `day` images side by side.

### Expected Output

- Grayscale and RGB images before and after histogram matching.




