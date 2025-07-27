using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        try
        {
            // Absolute paths for reliability
            string baseDir = "/Users/reveal/Documents/ImageClassifier/";
            string modelPath = Path.Combine(baseDir, "Models/squeezenet1.1-7.onnx");
            string imagesFolder = Path.Combine(baseDir, "Images");
            string labelsPath = Path.Combine(baseDir, "Models/labels.txt");

            // Enhanced validation
            ValidateFile(modelPath, "ONNX model", minimumSize: 4_800_000); // ~4.8MB
            ValidateFile(labelsPath, "labels file");
            ValidateDirectory(imagesFolder);

            // Get image to process
            string imageFile = GetImageToProcess(imagesFolder, args);

            // Initialize MLContext
            var mlContext = new MLContext();

            // Build pipeline with explicit column names
            var pipeline = mlContext.Transforms.LoadImages(
                    outputColumnName: "image_tensor",
                    imageFolder: "",
                    inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(
                    outputColumnName: "resized_image",
                    imageWidth: 224,
                    imageHeight: 224,
                    inputColumnName: "image_tensor"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "data_0",
                    inputColumnName: "resized_image"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    modelFile: modelPath,
                    outputColumnNames: new[] { "softmaxout_1" },
                    inputColumnNames: new[] { "data_0" }));

            // Create and fit model
            var data = mlContext.Data.LoadFromEnumerable(new[] { new ImageData { ImagePath = imageFile } });
            var model = pipeline.Fit(data);

            // Make prediction
            var engine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);
            var prediction = engine.Predict(new ImageData { ImagePath = imageFile });

            // Display results
            var labels = File.ReadAllLines(labelsPath);
            var scores = Softmax(prediction.PredictedLabels);

            Console.WriteLine("\nTop Predictions:");
            scores.Select((score, i) => (Label: labels[i], Score: score))
                  .OrderByDescending(x => x.Score)
                  .Take(5)
                  .ToList()
                  .ForEach(x => Console.WriteLine($"{x.Label}: {x.Score:P2}"));
        }
        catch (Exception ex)
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"Error: {ex.GetType().Name}: {ex.Message}");
            Console.ResetColor();
            Console.WriteLine("\nTroubleshooting Checklist:");
            Console.WriteLine("1. Model file exists and is valid (try the verification script)");
            Console.WriteLine("2. All paths are correct and accessible");
            Console.WriteLine("3. You have read permissions for all files");
            Console.WriteLine("4. The model matches the expected architecture");
        }
    }

    static void ValidateFile(string path, string description, int? minimumSize = null)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"{description} not found at {path}");

        if (minimumSize.HasValue && new FileInfo(path).Length < minimumSize)
            throw new InvalidDataException($"{description} is too small (possibly corrupted)");
    }

    static void ValidateDirectory(string path)
    {
        if (!Directory.Exists(path))
            throw new DirectoryNotFoundException($"Directory not found at {path}");

        if (!Directory.EnumerateFiles(path).Any())
            throw new InvalidOperationException($"No files found in {path}");
    }

    static string GetImageToProcess(string imagesFolder, string[] args)
    {
        if (args.Length > 0 && File.Exists(args[0]))
            return args[0];

        var image = Directory.EnumerateFiles(imagesFolder)
            .FirstOrDefault(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                               f.EndsWith(".png", StringComparison.OrdinalIgnoreCase));

        return image ?? throw new FileNotFoundException("No JPG/PNG images found in directory");
    }

    static float[] Softmax(float[] values)
    {
        var max = values.Max();
        var exp = values.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    public class ImageData
    {
        public string ImagePath { get; set; }
    }

    public class ImagePrediction
    {
        [ColumnName("softmaxout_1")]
        public float[] PredictedLabels { get; set; }
    }
}
