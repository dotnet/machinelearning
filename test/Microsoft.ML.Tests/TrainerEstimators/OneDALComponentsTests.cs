using System.Collections.Generic;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.TestFramework.Attributes;
using Xunit;

namespace Microsoft.ML.Tests.TrainerEstimators
{
    public partial class TrainerEstimators
    {
	internal class DataPoint1
	{
	   public float Label { get; set; }
           [VectorType(1)]
	   public float[] Features { get; set; }
	}

	// [NativeDependencyFact("OneDALNative")]
	[Fact]
	public void TestEstimatorOneDALLinReg()
        {
	      List<DataPoint1> literalData = new List<DataPoint1>
	      {
	          new DataPoint1 { Features = new float[]{1},    Label= 39000 },
	          new DataPoint1 { Features = new float[]{1.3F}, Label= 46200 },
		  new DataPoint1 { Features = new float[]{1.5F}, Label= 37700 },
		  new DataPoint1 { Features = new float[]{2},    Label= 43500 },
		  new DataPoint1 { Features = new float[]{2.2F}, Label= 40000 },
		  new DataPoint1 { Features = new float[]{2.9F}, Label= 56000 }
	      };

	      var dataView = ML.Data.LoadFromEnumerable(literalData);

//            TestEstimatorCore(trainer, dataView);

	      var model = trainer.Fit(dataView);
	      var modelParameters = ((ISingleFeaturePredictionTransformer<object>)model).Model as LinearRegressionModelParameters;
//            Assert.True(model.Model.HasStatistics);
//            Assert.NotEmpty(model.Model.StandardErrors);
//            Assert.NotEmpty(model.Model.PValues);
//            Assert.NotEmpty(model.Model.TValues);
	      var retrainedModel = ML.Regression.Trainers.OnlineGradientDescent(numberOfIterations: 1).Fit(dataView, modelParameters);

//	    Assert.True(File.Exists("libOneDALNative.so"));
	    Assert.False(trainer.Info.WantCaching);
            Done();
        }
    }
}
