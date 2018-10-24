// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime.Data;

namespace Microsoft.ML.Runtime.Training
{
    /// <summary>
    /// This represents a basic class for 'simple trainer'.
    /// A 'simple trainer' accepts one feature column and one label column, also optionally a weight column.
    /// It produces a 'prediction transformer'.
    /// </summary>
    public abstract class TrainerEstimatorBase<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>, ITrainer<TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : IPredictor
    {
        /// <summary>
        /// A standard string to use in errors or warnings by subclasses, to communicate the idea that no valid
        /// instances were able to be found.
        /// </summary>
        protected const string NoTrainingInstancesMessage = "No valid training instances found, all instances have missing features.";

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

        /// <summary>
        /// The optional groupID column that the ranking trainers expects.
        /// </summary>
        public readonly SchemaShape.Column GroupIdColumn;

        protected readonly IHost Host;

        /// <summary>
        /// The information about the trainer: whether it benefits from normalization, caching etc.
        /// </summary>
        public abstract TrainerInfo Info { get; }

        public abstract PredictionKind PredictionKind { get; }

        public TrainerEstimatorBase(IHost host,
            SchemaShape.Column feature,
            SchemaShape.Column label,
            SchemaShape.Column weight = null,
            SchemaShape.Column groupId = null)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
            Host.CheckValue(feature, nameof(feature));
            Host.CheckValueOrNull(label);
            Host.CheckValueOrNull(weight);
            Host.CheckValueOrNull(groupId);

            FeatureColumn = feature;
            LabelColumn = label;
            WeightColumn = weight;
            GroupIdColumn = groupId;
        }

        public TTransformer Fit(IDataView input) => TrainTransformer(input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.Columns.ToDictionary(x => x.Name);
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        /// <summary>
        /// The columns that will be created by the fitted transformer.
        /// </summary>
        protected abstract SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema);

        public TModel Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            return TrainModelCore(context);
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(FeatureColumn.Name, out var featureCol))
                throw Host.Except($"Feature column '{FeatureColumn.Name}' is not found");
            if (!FeatureColumn.IsCompatibleWith(featureCol))
                throw Host.Except($"Feature column '{FeatureColumn.Name}' is not compatible");

            if (WeightColumn != null)
            {
                if (!inputSchema.TryFindColumn(WeightColumn.Name, out var weightCol))
                    throw Host.Except($"Weight column '{WeightColumn.Name}' is not found");
                if (!WeightColumn.IsCompatibleWith(weightCol))
                    throw Host.Except($"Weight column '{WeightColumn.Name}' is not compatible");
            }

            // Special treatment for label column: we allow different types of labels, so the trainers
            // may define their own requirements on the label column.
            if (LabelColumn != null)
            {
                if (!inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol))
                    throw Host.Except($"Label column '{LabelColumn.Name}' is not found");
                CheckLabelCompatible(labelCol);
            }
        }

        protected virtual void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.CheckValue(labelCol, nameof(labelCol));
            Contracts.AssertValue(LabelColumn);

            if (!LabelColumn.IsCompatibleWith(labelCol))
                throw Host.Except($"Label column '{LabelColumn.Name}' is not compatible");
        }

        protected TTransformer TrainTransformer(IDataView trainSet,
            IDataView validationSet = null, IPredictor initPredictor = null)
        {
            var cachedTrain = Info.WantCaching ? new CacheDataView(Host, trainSet, prefetch: null) : trainSet;

            var trainRoles = MakeRoles(cachedTrain);

            RoleMappedData validRoles;

            if (validationSet == null)
                validRoles = null;
            else
            {
                var cachedValid = Info.WantCaching ? new CacheDataView(Host, validationSet, prefetch: null) : validationSet;
                validRoles = MakeRoles(cachedValid);
            }

            var pred = TrainModelCore(new TrainContext(trainRoles, validRoles, initPredictor));
            return MakeTransformer(pred, trainSet.Schema);
        }

        protected abstract TModel TrainModelCore(TrainContext trainContext);

        protected abstract TTransformer MakeTransformer(TModel model, Schema trainSchema);

        private RoleMappedData MakeRoles(IDataView data) =>
            new RoleMappedData(data, label: LabelColumn?.Name, feature: FeatureColumn.Name, group: GroupIdColumn?.Name, weight: WeightColumn?.Name);

        IPredictor ITrainer.Train(TrainContext context) => Train(context);
    }
}
