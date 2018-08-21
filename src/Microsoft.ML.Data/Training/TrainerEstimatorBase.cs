using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Training
{
    /// <summary>
    /// This represents a basic class for 'simple trainer'.
    /// A 'simple trainer' accepts one feature column and one label column, also optionally a weight column.
    /// It produces a 'prediction transformer'.
    /// </summary>
    /// <typeparam name="TTransformer"></typeparam>
    /// <typeparam name="TModel"></typeparam>
    public abstract class TrainerEstimatorBase<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>
        where TTransformer : IPredictionTransformer<TModel>
        where TModel : IPredictor
    {
        /// <summary>
        /// The feature column that the trainer expects.
        /// </summary>
        public readonly SchemaShape.Column FeatureColumn;

        /// <summary>
        /// The label column that the trainer expects. Can be <c>null</c>, which indicates that label
        /// is not used for training.
        /// </summary>
        public readonly SchemaShape.Column LabelColumn;

        /// <summary>
        /// The weight column that the trainer expects. Can be <c>null</c>, which indicates that weight is
        /// not used for training.
        /// </summary>
        public readonly SchemaShape.Column WeightColumn;

        protected readonly IHost Host;

        /// <summary>
        /// The information about the trainer: whether it benefits from normalization, caching etc.
        /// </summary>
        public TrainerInfo TrainerInfo { get; }

        public TrainerEstimatorBase(IHost host,
            SchemaShape.Column feature, SchemaShape.Column label, SchemaShape.Column weight = null)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
            Host.CheckValue(feature, nameof(feature));
            Host.CheckValueOrNull(label);
            Host.CheckValueOrNull(weight);

            FeatureColumn = feature;
            LabelColumn = label;
            WeightColumn = weight;
        }

        protected TTransformer TrainTransformer(IDataView trainSet,
            IDataView validationSet = null, IPredictor initPredictor = null)
        {
            var cachedTrain = TrainerInfo.WantCaching ? new CacheDataView(Host, trainSet, prefetch: null) : trainSet;

            var trainRoles = MakeRoles(cachedTrain);

            RoleMappedData validRoles;

            if (validationSet == null)
                validRoles = null;
            else
            {
                var cachedValid = TrainerInfo.WantCaching ? new CacheDataView(Host, validationSet, prefetch: null) : validationSet;
                validRoles = MakeRoles(cachedValid);
            }

            var pred = TrainModelCore(new TrainContext(trainRoles, validRoles, initPredictor));

            var emptyData = new EmptyDataView(Host, trainSet.Schema);
            var scoreRoles = MakeRoles(emptyData);
            return MakeTransformer(pred);
        }

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            throw new NotImplementedException();
        }

        protected abstract TModel TrainModelCore(TrainContext trainContext);

        protected abstract TTransformer MakeTransformer(TModel model);

        private RoleMappedData MakeRoles(IDataView data) =>
            new RoleMappedData(data, label: LabelColumn.Name, feature: FeatureColumn.Name, weight: WeightColumn.Name);
    }
}
