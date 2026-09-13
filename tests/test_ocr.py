import unittest
from PIL import Image
from nano_ocr import NanoOCR, OCRResult

class TestNanoOCR(unittest.TestCase):
    def setUp(self):
        # Initialize the engine (tests if weights load correctly)
        self.ocr = NanoOCR()
        # Create a blank 128x32 image in memory to test the pipeline
        self.dummy_image = Image.new('L', (128, 32), color=0)

    def test_read_returns_string(self):
        """Ensure the read() function outputs a Python string."""
        text = self.ocr.read(self.dummy_image)
        self.assertIsInstance(text, str)

    def test_predict_returns_result_object(self):
        """Ensure predict() outputs the structured OCRResult with metrics."""
        result = self.ocr.predict(self.dummy_image)
        self.assertIsInstance(result, OCRResult)
        self.assertIsInstance(result.text, str)
        self.assertIsInstance(result.confidence, float)
        self.assertIsInstance(result.inference_time, float)

if __name__ == '__main__':
    unittest.main()