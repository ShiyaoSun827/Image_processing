# Advanced JPEG Compression and Decompression

## Overview: High-Efficiency JPEG Compression and Decompression

This project implements a state-of-the-art **JPEG encoding and decoding pipeline**, leveraging advanced signal processing techniques to achieve high-compression efficiency while minimizing perceptual loss. By employing frequency-domain transformation, quantization, and entropy encoding, this implementation optimizes image storage and transmission in computer vision, media processing, and machine learning applications.

## Implementation Details

### 1. Color Space Transformation
- Convert the input **RGB image to YCbCr color space**, separating luminance and chrominance components for more efficient compression.
- Utilize matrix transformations for precision and computational efficiency.

### 2. Block-Based Discrete Cosine Transform (DCT)
- Segment the YCbCr image into **8x8 pixel blocks** for localized frequency analysis.
- Apply **2D Discrete Cosine Transform (DCT)** using **scipy.fftpack.dct** to convert spatial information into frequency components.
- Ensure optimal handling of low and high-frequency coefficients to maximize compression gains.

### 3. Quantization for Data Compression
- Implement **standard JPEG quantization matrices** to selectively reduce high-frequency details.
- Compute the **compression ratio** by analyzing zero-value coefficients in quantized blocks.
- Balance quality and compression through adaptive quantization techniques.

### 4. Zig-Zag Reordering and Run-Length Encoding
- Reorder quantized coefficients using **zig-zag scanning** for more efficient entropy encoding.
- Apply **Run-Length Encoding (RLE)** to compactly represent sequences of zeroes.

### 5. Huffman Entropy Encoding
- Implement **Huffman coding** to further compress the encoded data.
- Leverage statistical redundancy in quantized coefficients to minimize file size.

### 6. JPEG Decoding Pipeline
- Reverse the process: **Inverse Huffman Decoding → Run-Length Decoding → Dequantization → Inverse DCT → YCbCr to RGB conversion**.
- Reconstruct the image with minimal perceptual loss, closely approximating the original.

## Execution

Run the script with:
```bash
python main.py
```
This will perform both encoding and decoding, displaying the reconstructed image and compression statistics.

## Applications and Use Cases

- **Image Compression for Storage and Transmission**: Reducing file sizes for efficient data storage and faster web image loading.
- **Medical and Satellite Imaging**: Compressing high-resolution images while preserving crucial visual information.
- **Machine Learning and AI**: Preprocessing large image datasets efficiently without excessive loss of detail.
- **Video Encoding**: Foundation for video codecs such as MPEG and H.264.

## References

- [JPEG Compression - Wikipedia](https://en.wikipedia.org/wiki/JPEG)
- [Discrete Cosine Transform - SciPy Docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html)
- [Lossy Image Compression - Stanford](https://cs.stanford.edu/people/eroberts/courses/soco/projects/data-compression/lossy/jpeg/coeff.htm)

This project showcases expertise in **high-performance image processing**, implementing industry-standard compression techniques with optimized efficiency and scalability.

