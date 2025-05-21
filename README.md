# 🗣️ Vosk ASR to RKNN Conversion for RK3588

This repository provides scripts and instructions to convert [Vosk](https://github.com/alphacep/vosk-api) speech recognition models (in ONNX format) into RKNN format, optimized for the RK3588 NPU.

Use cases include real-time offline transcription on devices such as Orange Pi 5 and Radxa Rock 5B.

---

## 🎯 Purpose

- Convert Vosk ONNX models to RKNN
- Enable low-latency ASR using the RK3588 NPU
- Avoid heavy CPU usage during audio transcription

---

## 🧱 Dependencies

- Python 3.8–3.10
- RKNN Toolkit 2
- `onnx==1.14.1`
- `numpy`
- Vosk ONNX models (`encoder.onnx`, `decoder.onnx`, `joiner.onnx`)

### 🔧 Install Python dependencies:

```bash
pip install onnx numpy
