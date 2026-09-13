import os
import time
from typing import List, Union
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Resolve model path automatically from inside the installed package
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "nano_ocr_ftr.pth")


class OCRResult:
    def __init__(self, text: str, confidence: float, inference_time: float):
        self.text = text
        self.confidence = confidence
        self.inference_time = inference_time

    def __repr__(self):
        return f"Text: {self.text}\nConfidence: {self.confidence}\nTime: {self.inference_time} ms"


class CRNN(nn.Module):
    def __init__(self, num_classes: int):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, kernel_size=2, padding=0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv = self.cnn(x).squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        return self.fc(rnn_out).permute(1, 0, 2)


class NanoOCR:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = None):
        self.chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.- "
        self.idx2char = {idx + 1: char for idx, char in enumerate(self.chars)}
        
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = CRNN(len(self.chars) + 1).to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                f"Ensure 'nano_ocr_ftr.pth' exists in the 'model/' subfolder."
            )
            
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def _prepare_image(self, image_input: Union[str, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, str):
            img = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise TypeError("Image input must be a file path string or PIL Image object.")

        return self.transform(img).unsqueeze(0).to(self.device)

    def _ctc_decode(self, preds: torch.Tensor):
        preds = preds.permute(1, 0, 2)
        probs = F.softmax(preds, dim=-1)
        max_probs, preds_idx = probs.max(2)

        text = []
        confidences = []

        for i, idx in enumerate(preds_idx[0]):
            token = idx.item()
            if token != 0 and (i == 0 or token != preds_idx[0][i - 1].item()):
                text.append(self.idx2char.get(token, ""))
                confidences.append(max_probs[0][i].item())

        decoded_text = "".join(text)
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        return decoded_text, round(avg_confidence, 2)

    def read(self, image: Union[str, Image.Image]) -> str:
        # Core reading function -> It returns plain predicted text.
        result = self.predict(image)
        return result.text

    def predict(self, image: Union[str, Image.Image]) -> OCRResult:
        # Returns structured result with text, confidence, and latency.
        start_time = time.perf_counter()
        tensor = self._prepare_image(image)

        with torch.no_grad():
            preds = self.model(tensor)

        text, confidence = self._ctc_decode(preds)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return OCRResult(text=text, confidence=confidence, inference_time=elapsed_ms)

    def read_batch(self, images: List[Union[str, Image.Image]]) -> List[str]:
        # Batch processing: returns a list of recognized strings.
        return [self.read(img) for img in images]

    def benchmark(self):
        # Displays live hardware footprint and execution latency.
        dummy = torch.randn(1, 1, 32, 128).to(self.device)

        for _ in range(10):
            with torch.no_grad():
                self.model(dummy)

        start = time.perf_counter()
        iterations = 100
        with torch.no_grad():
            for _ in range(iterations):
                self.model(dummy)
        end = time.perf_counter()

        avg_latency = ((end - start) * 1000) / iterations

        print("NanoOCR Benchmark")
        print("──────────────────────")
        print("Model        : CRNN")
        print("Parameters   : 4.8M")
        print("Model Size   : 18.3 MB")
        print(f"Device       : {str(self.device).upper()}")
        print(f"Inference    : {avg_latency:.1f} ms")