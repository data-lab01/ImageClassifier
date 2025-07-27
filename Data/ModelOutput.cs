using Microsoft.ML.Data;

namespace ImageClassifier.Data
{
    public class ModelOutput : ImageData
    {
        [ColumnName("PredictedLabel")]
        public string PredictedLabel { get; set; } = string.Empty;

        public float[]? Score { get; set; }
    }
}
