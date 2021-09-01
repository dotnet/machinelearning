using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Microsoft.ML
{
    public static class OneDALCatalog
    {
<<<<<<< HEAD
	public static LinRegTrainer LinReg(this RegressionCatalog.RegressionTrainers catalog,
               	      		    	   string labelColumnName = DefaultColumnNames.Label,
					   string featureColumnName = DefaultColumnNames.Features,
					   string exampleWeightColumnName = null)
	{
=======
        public static LinRegTrainer LinReg(this RegressionCatalog.RegressionTrainers catalog,
            string labelColumnName = DefaultColumnNames.Label,
            string featureColumnName = DefaultColumnNames.Features,
            string exampleWeightColumnName = null)
        {
>>>>>>> cce1bc84 (Initial integration for oneDAL LinearRegression)
            Contracts.CheckValue(catalog, nameof(catalog));
            var env = CatalogUtils.GetEnvironment(catalog);
            var options = new LinRegTrainer.Options
            {
                LabelColumnName = labelColumnName,
                FeatureColumnName = featureColumnName,
                ExampleWeightColumnName = exampleWeightColumnName
            };
<<<<<<< HEAD
 
            return new LinRegTrainer(env, options);
	}

	public static LinRegTrainer LinReg(this RegressionCatalog.RegressionTrainers catalog,
	       	      		           LinRegTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));
 
=======

            return new LinRegTrainer(env, options);
        }

        public static LinRegTrainer LinReg(
            this RegressionCatalog.RegressionTrainers catalog,
            LinRegTrainer.Options options)
        {
            Contracts.CheckValue(catalog, nameof(catalog));
            Contracts.CheckValue(options, nameof(options));

>>>>>>> cce1bc84 (Initial integration for oneDAL LinearRegression)
            var env = CatalogUtils.GetEnvironment(catalog);
            return new LinRegTrainer(env, options);
        }
    }
}
