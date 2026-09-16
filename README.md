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

## Evaluation Metrics

### Dataset Accuracy
- ICDAR 2013: 81.28%
- MJSynth (Unseen): 79.20%
- IIIT5k: 78.37%
- ICDAR 2015 (Heavy noise): 46.41%

### Error Metrics
- CTC Loss (ICDAR '13): 0.44
- CER (ICDAR '13): 6.73%
- CER (IIIT5k): 8.56%

*CER (Character Error Rate) indicates that even when an entire word is predicted "wrong", the model typically only misses a single character.*

### Error Typology (15,000 Character Sample)
`602 Substitutions (e.g. '0' vs 'O')`
`628 Deletions (Missing thin letters)`
`77 Insertions (Hallucinations)`

Confidence Calibration: The engine is self-aware. It averages 95.7% confidence on correct predictions, and drops to 76.7% when incorrect, allowing developers to build safe failure thresholds.

## Author
Developed by Arshvir
