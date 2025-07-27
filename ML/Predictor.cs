using Microsoft.ML;
using ImageClassifier.Data;

namespace ImageClassifier.ML
{
    public class Predictor
    {
        private readonly PredictionEngine<ImageData, ModelOutput> predictor;

        public Predictor(string modelPath)
        {
            var mlContext = new MLContext();
            var model = mlContext.Model.Load(modelPath, out _);
            predictor = mlContext.Model.CreatePredictionEngine<ImageData, ModelOutput>(model);
        }

        public void Predict(string imagePath)
        {
            var input = new ImageData { ImagePath = imagePath };
            var prediction = predictor.Predict(input);
            Console.WriteLine($"Prediction: {prediction.PredictedLabel}");
        }
    }
}
