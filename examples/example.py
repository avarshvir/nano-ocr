import os
from PIL import Image, ImageDraw
from nano_ocr import NanoOCR

def create_dummy_image(path="test_image.png"):
    #Creates a simple 128x32 black image with a white line to test the engine
    img = Image.new('L', (128, 32), color=0)
    draw = ImageDraw.Draw(img)
    draw.line((20, 16, 108, 16), fill=255, width=2)
    img.save(path)
    return path

def main():
    print("Initializing NanoOCR Engine...")
    ocr = NanoOCR()

    # 1. Create a test image
    img_path = create_dummy_image()
    print(f"\nCreated test image at: {img_path}")

    # 2. Test the standard read() function
    print("\n--- Testing read() ---")
    text = ocr.read(img_path)
    print(f"Extracted Text: {text}")

    # 3. Test the detailed predict() function
    print("\n--- Testing predict() ---")
    result = ocr.predict(img_path)
    print(result)

    # 4. Run the hardware benchmark
    print("\n--- Running System Benchmark ---")
    ocr.benchmark()

    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)

if __name__ == "__main__":
    main()