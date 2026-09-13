# NanoOCR 🚀

A lightning-fast, production-ready Convolutional Recurrent Neural Network (CRNN) OCR engine optimized for edge devices.

Built from scratch, trained on over 2.3 million synthetic and real-world images, and packaged into a tiny **18.3 MB footprint**.

## Features
- **Ultra-Lightweight:** 4.8M parameters, under 20 MB on disk.
- **Blazing Fast:** 1.74 ms latency (~575 FPS) on a T4 GPU.
- **Dual-Domain Mastery:** Bridged the gap between synthetic data (MJSynth) and messy real-world photography (IIIT5K).
- **High Accuracy:** 78.37% exact match and **8.56% Character Error Rate (CER)** on real-world scene text.

## Installation
`pip install nano-ocr`

## Quick Start
```
from nano_ocr import NanoOCR

# Initialize the engine (automatically loads the bundled 18MB ftr weights)
ocr = NanoOCR()

# 1. Standard Read
text = ocr.read("image.png")
print(text)
# Output: Hello World

# 2. Detailed Prediction (Confidence & Latency)
result = ocr.predict("image.png")
print(result)
# Output:
# Text: Hello World
# Confidence: 0.94
# Time: 31.2 ms

# 3. Batch Processing
images = ["img1.png", "img2.png", "img3.png"]
results = ocr.read_batch(images)
print(results)

# 4. System Hardware Benchmark
ocr.benchmark()
```

## Architecture
NanoOCR utilizes a 7-layer VGG-style CNN for feature extraction (with asymmetric max-pooling), a Bidirectional LSTM (256 hidden size) for sequence context, and CTC loss for decoding.

## Author
Developed by Arshvir
