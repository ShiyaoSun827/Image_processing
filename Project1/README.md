# Advanced Histogram Processing Techniques

## Overview
This project implements sophisticated image processing techniques focused on histogram computation, equalization, comparison, and matching. These methods are essential in computer vision, medical imaging, and machine learning applications, showcasing expertise in low-level image processing and statistical analysis.

## Key Technologies and Techniques
- **Manual Histogram Computation**: Directly computing image histograms without relying on built-in functions, demonstrating a deep understanding of pixel intensity distributions.
- **Histogram Equalization**: Implementing contrast enhancement using cumulative distribution function (CDF) transformations, significantly improving image clarity.
- **Histogram Comparison**: Utilizing **Bhattacharyya Coefficient** for quantitative similarity measurement between histograms, a technique widely used in pattern recognition and statistical inference.
- **Histogram Matching**: Executing precise histogram transformation techniques to modify one image’s distribution to match another’s, an advanced concept applied in photometric normalization and style transfer.
- **RGB and Grayscale Processing**: Handling both color and monochrome images, applying transformations per channel, and combining results for realistic visual adjustments.
- **Matplotlib for Data Visualization**: Effectively plotting and visualizing histograms, equalized images, and matched images for comparative analysis.
- **NumPy for Efficient Computation**: Leveraging NumPy for matrix operations and histogram calculations, ensuring optimal performance and scalability.

## Implementation Details
### 1. Histogram Computation
- Manually compute a 64-bin grayscale histogram from pixel intensities.
- Compare with NumPy’s histogram function to validate accuracy.

### 2. Histogram Equalization
- Compute the image’s cumulative histogram and apply equalization.
- Improve contrast by redistributing intensity levels.

### 3. Histogram Comparison
- Convert images to grayscale and compute 256-bin histograms.
- Normalize histograms and calculate the Bhattacharyya Coefficient.

### 4. Histogram Matching
#### **Grayscale Matching:**
- Adjust the pixel intensity mapping of one image to match another.
- Utilize CDF-based transformations to achieve smooth transitions.

#### **RGB Matching:**
- Perform per-channel histogram transformations in the RGB space.
- Seamlessly integrate results for a perceptually coherent image transformation.

## Execution
Run the script with:
```bash
python A1_submission.py
```
This will sequentially execute all processing techniques, delivering results in visual plots and computed metrics.

## Applications and Use Cases
- **Medical Image Processing**: Enhancing contrast in medical scans such as X-rays and MRIs.
- **Computer Vision**: Preprocessing images for better feature extraction in deep learning models.
- **Photography and Graphics**: Correcting exposure and adapting color distributions in image enhancement workflows.
- **Security and Surveillance**: Improving low-light images for better clarity in CCTV footage.

## References
- [Histogram Equalization - Wikipedia](https://en.wikipedia.org/wiki/Histogram_equalization)
- [Bhattacharyya Distance - Wikipedia](https://en.wikipedia.org/wiki/Bhattacharyya_distance)
- [Histogram Matching - Towards Data Science](https://towardsdatascience.com/histogram-matching-ee3a67b4cbc1)

This project exemplifies expertise in fundamental and advanced image processing techniques, combining computational efficiency with practical applications in real-world scenarios.

