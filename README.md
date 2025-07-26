<img width="1160" height="1160" alt="image" src="https://github.com/user-attachments/assets/0af763d5-d8a4-4753-84df-98ec4a273b32" />


# 🧠 Image Classifier using ONNX and ML.NET

This is a simple image classification app built with **.NET**, **ML.NET**, and a **pre-trained ONNX model (SqueezeNet)**. 
It loads an image, runs it through the model, and outputs the top predicted label.

---

## 📸 Example Output

```
Prediction: cat (94.32%)
```

## ✅ Features
- Load and classify any image
- Run predictions using a pre-trained ONNX model
- Display top result and confidence score
- Easily replace the model with another ONNX file

## 🛠 Requirements
- [.NET 8 SDK](https://dotnet.microsoft.com/download)

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/data-lab01/ImageClassifier.git
cd ImageClassifier
```

### 2. Download ONNX Model & Labels
- Place `squeezenet1.1-7.onnx` into `Models/`
- Download `labels.txt` from [here](https://raw.githubusercontent.com/onnx/models/main/vision/classification/squeezenet/model/labels.txt)

### 3. Add an Image
Put your test image in the `Images/` folder and name it `umu.jpg` (or edit the path in `Program.cs`)

### 4. Run the App
```bash
dotnet run
```

## 📘 References
- [ML.NET Documentation](https://learn.microsoft.com/en-us/dotnet/machine-learning/)
- [ONNX Models Zoo](https://github.com/onnx/models)

## 📄 License
MIT License

## 🙋 Author
**Robert Bakyayita**  
https://www.linkedin.com/in/robertbakyayita1/
