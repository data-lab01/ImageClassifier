using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;

class Program
{
    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        var modelPath = "Models/squeezenet1.1-7.onnx";
        var imagePath = "Images/sample.jpg";
        var labelsPath = "Models/labels.txt";

        var data = new[] { new ImageInputData { ImagePath = imagePath } };
        var imageData = mlContext.Data.LoadFromEnumerable(data);

        var pipeline = mlContext.Transforms.LoadImages("input", ".", nameof(ImageInputData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages("input", 224, 224, "input"))
            .Append(mlContext.Transforms.ExtractPixels("input"))
            .Append(mlContext.Transforms.ApplyOnnxModel(
                modelFile: modelPath,
                outputColumnNames: new[] { "squeezenet0_flatten0_reshape0" },
                inputColumnNames: new[] { "input" }));

        var model = pipeline.Fit(imageData);
        var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageInputData, ImagePrediction>(model);

        var prediction = predictionEngine.Predict(new ImageInputData { ImagePath = imagePath });

        var labels = File.ReadAllLines(labelsPath);
        var maxScore = float.MinValue;
        var maxIndex = -1;

        for (int i = 0; i < prediction.PredictedLabels.Length; i++)
        {
            if (prediction.PredictedLabels[i] > maxScore)
            {
                maxScore = prediction.PredictedLabels[i];
                maxIndex = i;
            }
        }

        Console.WriteLine($"Prediction: {labels[maxIndex]} ({maxScore:P2})");
    }

    class ImageInputData
    {
        public string ImagePath { get; set; }
    }

    class ImagePrediction
    {
        [ColumnName("squeezenet0_flatten0_reshape0")]
        public float[] PredictedLabels { get; set; }
    }
}
