# Advanced Image Segmentation and Object Detection

## Overview: Computational Cell Analysis ,Image Segmentation and Object Detection

This project implements state-of-the-art image segmentation techniques using advanced blob detection, thresholding, and watershed algorithms. The system efficiently detects and segments objects within an image, leveraging computational imaging techniques that are widely applicable in biomedical image analysis, object recognition, and automated microscopy.

## Implementation Details

### 1. Difference-of-Gaussian (DoG) Blob Detection
- Construct a multi-scale **Difference-of-Gaussian (DoG) volume** to detect image structures at different scales.
- Utilize Gaussian filtering via **skimage.filters.gaussian**, ensuring precise noise reduction.
- Compute the DoG response by subtracting two blurred images at varying sigma values.

### 2. Regional Minima Detection for Blob Localization
- Implement **regional minima detection** to estimate initial blob center locations.
- Utilize **scipy.ndimage.filters.minimum_filter** to identify low-intensity regions.
- Convert detected minima into a binary mask, overlaying detected locations onto the original image.

### 3. Li’s Adaptive Thresholding for Blob Refinement
- Apply **Li’s thresholding** using **skimage.filters.threshold_li** to refine blob detection.
- Filter out spurious minima that fall below the threshold.
- Overlay refined blob centers onto the image for visualization.

### 4. Watershed Segmentation for Object Boundary Detection
- Implement a **Minimum Following Watershed Algorithm** for precise cell boundary detection.
- Compute the **image gradient magnitude** as a preliminary step in segmentation.
- Employ **iterative pixel propagation** to assign each pixel to the nearest local minimum.
- Handle **4-connected neighborhood constraints** to maintain region consistency.

## Execution

Run the script with:
```bash
python main.py
```
This will execute all segmentation processes sequentially, generating visual overlays of detected blobs and segmented regions.

## Applications and Use Cases

- **Biomedical Imaging**: Automated cell segmentation in microscopic images.
- **Autonomous Systems**: Object detection and recognition in real-time vision systems.
- **Satellite Image Analysis**: Identifying geographical structures and landforms.
- **Quality Control in Manufacturing**: Detecting defects and irregularities in industrial inspection.

## References

- [Difference of Gaussians - Wikipedia](https://en.wikipedia.org/wiki/Difference_of_Gaussians)
- [Watershed Segmentation - MATLAB](https://www.mathworks.com/help/images/ref/imregionalmin.html)
- [Li’s Thresholding - Scikit-Image Docs](https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_li)

This project showcases expertise in advanced image segmentation techniques, integrating computational efficiency with real-world applications in scientific and industrial imaging.

