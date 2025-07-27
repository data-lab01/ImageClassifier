using ImageClassifier.ML;

class Program
{
    static void Main(string[] args)
    {
        string datasetPath = "dataset";
        string modelPath = "model.zip";
        string testImagePath = "/Users/reveal/Documents/ImageClassifier/dataset/cats/cat.jpg"; // replace this

        var builder = new ModelBuilder();
        var model = builder.Train(datasetPath, out var schema);
        builder.SaveModel(model, schema, modelPath);

        var predictor = new Predictor(modelPath);
        predictor.Predict(testImagePath);
    }
}
