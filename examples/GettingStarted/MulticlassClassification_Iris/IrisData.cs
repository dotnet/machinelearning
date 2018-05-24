using Microsoft.ML.Runtime.Api;

namespace MulticlassClassification_Iris
{
    public class IrisData
    {
        [Column("0")]
        public float Label;

        [Column("1")]
        public float SepalLength;

        [Column("2")]
        public float SepalWidth;

        [Column("3")]
        public float PetalLength;

        [Column("4")]
        public float PetalWidth;
    }
}