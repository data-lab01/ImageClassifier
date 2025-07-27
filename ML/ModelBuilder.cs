using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Vision;
using ImageClassifier.Data;

namespace ImageClassifier.ML
{
    public class ModelBuilder
    {
        private readonly MLContext mlContext = new();

        public ITransformer Train(string datasetPath, out DataViewSchema schema)
        {
            var data = Directory.GetFiles(datasetPath, "*", SearchOption.AllDirectories)
                .Select(path => new ImageData
                {
                    ImagePath = path,
                    Label = Path.GetFileName(Path.GetDirectoryName(path)!)
                });

            var imageData = mlContext.Data.LoadFromEnumerable(data);
            var split = mlContext.Data.TrainTestSplit(imageData, testFraction: 0.2);

            var pipeline = mlContext.Transforms.LoadRawImageBytes(
                    outputColumnName: "Image",
                    imageFolder: "",
                    inputColumnName: nameof(ImageData.ImagePath))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(mlContext.MulticlassClassification.Trainers.ImageClassification(
                    new ImageClassificationTrainer.Options
                    {
                        FeatureColumnName = "Image",
                        LabelColumnName = "Label",
                        Arch = ImageClassificationTrainer.Architecture.ResnetV2101,
                        MetricsCallback = metrics => Console.WriteLine($"LogLoss: {metrics.LogLoss}"),
                        ValidationSet = split.TestSet,
                        UseGpu = false
                    }))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            Console.WriteLine("Training...");
            var model = pipeline.Fit(split.TrainSet);
            Console.WriteLine("Training complete.");

            var predictions = model.Transform(split.TestSet);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy:P2}");
            Console.WriteLine($"MacroAccuracy: {metrics.MacroAccuracy:P2}");

            schema = imageData.Schema;
            return model;
        }

        public void SaveModel(ITransformer model, DataViewSchema schema, string modelPath)
        {
            mlContext.Model.Save(model, schema, modelPath);
            Console.WriteLine($"Model saved to: {modelPath}");
        }
    }
}
