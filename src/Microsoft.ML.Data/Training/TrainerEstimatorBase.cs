// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers
{
    /// <summary>
    /// This represents a basic class for 'simple trainer'.
    /// A 'simple trainer' accepts one feature column and one label column, also optionally a weight column.
    /// It produces a 'prediction transformer'.
    /// </summary>
    public abstract class TrainerEstimatorBase<TTransformer, TModel> : ITrainerEstimator<TTransformer, TModel>, ITrainer<IPredictor>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class
    {
        /// <summary>
        /// A standard string to use in errors or warnings by subclasses, to communicate the idea that no valid
        /// instances were able to be found.
        /// </summary>
        [BestFriend]
        private protected const string NoTrainingInstancesMessage = "No valid training instances found, all instances have missing features.";

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

        [BestFriend]
        private protected readonly IHost Host;

        /// <summary>
        /// The information about the trainer: whether it benefits from normalization, caching etc.
        /// </summary>
        public abstract TrainerInfo Info { get; }

        PredictionKind ITrainer.PredictionKind => PredictionKind;

        [BestFriend]
        private protected abstract PredictionKind PredictionKind { get; }

        [BestFriend]
        private protected TrainerEstimatorBase(IHost host,
            SchemaShape.Column feature,
            SchemaShape.Column label,
            SchemaShape.Column weight = default)
        {
            Contracts.CheckValue(host, nameof(host));
            Host = host;
            Host.CheckParam(feature.IsValid, nameof(feature), "not initialized properly");

            FeatureColumn = feature;
            LabelColumn = label;
            WeightColumn = weight;
        }

        /// <summary> Trains and returns a <see cref="ITransformer"/>.</summary>
        /// <remarks>
        /// Derived class can overload this function.
        /// For example, it could take an additional dataset to train with a separate validation set.
        /// </remarks>
        public TTransformer Fit(IDataView input) => TrainTransformer(input);

        public SchemaShape GetOutputSchema(SchemaShape inputSchema)
        {
            Host.CheckValue(inputSchema, nameof(inputSchema));

            CheckInputSchema(inputSchema);

            var outColumns = inputSchema.ToDictionary(x => x.Name);
            foreach (var col in GetOutputColumnsCore(inputSchema))
                outColumns[col.Name] = col;

            return new SchemaShape(outColumns.Values);
        }

        /// <summary>
        /// The columns that will be created by the fitted transformer.
        /// </summary>
        private protected abstract SchemaShape.Column[] GetOutputColumnsCore(SchemaShape inputSchema);

        IPredictor ITrainer<IPredictor>.Train(TrainContext context)
        {
            Host.CheckValue(context, nameof(context));
            var pred = TrainModelCore(context) as IPredictor;
            Host.Check(pred != null, "Training did not return a predictor.");
            return pred;
        }

        private void CheckInputSchema(SchemaShape inputSchema)
        {
            // Verify that all required input columns are present, and are of the same type.
            if (!inputSchema.TryFindColumn(FeatureColumn.Name, out var featureCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumn.Name);
            if (!FeatureColumn.IsCompatibleWith(featureCol))
                throw Host.ExceptSchemaMismatch(nameof(inputSchema), "feature", FeatureColumn.Name,
                    FeatureColumn.GetTypeString(), featureCol.GetTypeString());

            if (WeightColumn.IsValid)
            {
                if (!inputSchema.TryFindColumn(WeightColumn.Name, out var weightCol))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "weight", WeightColumn.Name);
                if (!WeightColumn.IsCompatibleWith(weightCol))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "weight", WeightColumn.Name,
                        WeightColumn.GetTypeString(), weightCol.GetTypeString());
            }

            // Special treatment for label column: we allow different types of labels, so the trainers
            // may define their own requirements on the label column.
            if (LabelColumn.IsValid)
            {
                if (!inputSchema.TryFindColumn(LabelColumn.Name, out var labelCol))
                    throw Host.ExceptSchemaMismatch(nameof(inputSchema), "label", LabelColumn.Name);
                CheckLabelCompatible(labelCol);
            }
        }

        private protected virtual void CheckLabelCompatible(SchemaShape.Column labelCol)
        {
            Contracts.CheckParam(labelCol.IsValid, nameof(labelCol), "not initialized properly");
            Host.Assert(LabelColumn.IsValid);

            if (!LabelColumn.IsCompatibleWith(labelCol))
                throw Host.ExceptSchemaMismatch(nameof(labelCol), "label", WeightColumn.Name,
                    LabelColumn.GetTypeString(), labelCol.GetTypeString());
        }

        [BestFriend]
        private protected TTransformer TrainTransformer(IDataView trainSet,
            IDataView validationSet = null, IPredictor initPredictor = null)
        {
            CheckInputSchema(SchemaShape.Create(trainSet.Schema));
            var trainRoleMapped = MakeRoles(trainSet);
            RoleMappedData validRoleMapped = null;

            if (validationSet != null)
            {
                CheckInputSchema(SchemaShape.Create(validationSet.Schema));
                validRoleMapped = MakeRoles(validationSet);
            }

            var pred = TrainModelCore(new TrainContext(trainRoleMapped, validRoleMapped, null, initPredictor));
            return MakeTransformer(pred, trainSet.Schema);
        }

        private protected abstract TModel TrainModelCore(TrainContext trainContext);

        private protected abstract TTransformer MakeTransformer(TModel model, DataViewSchema trainSchema);

        private protected virtual RoleMappedData MakeRoles(IDataView data) =>
            new RoleMappedData(data, label: LabelColumn.Name, feature: FeatureColumn.Name, weight: WeightColumn.Name);

        IPredictor ITrainer.Train(TrainContext context) => ((ITrainer<IPredictor>)this).Train(context);
    }

    /// <summary>
    /// This represents a basic class for 'simple trainer'.
    /// A 'simple trainer' accepts one feature column and one label column, also optionally a weight column.
    /// It produces a 'prediction transformer'.
    /// </summary>
    public abstract class TrainerEstimatorBaseWithGroupId<TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
        where TTransformer : ISingleFeaturePredictionTransformer<TModel>
        where TModel : class

    {
        /// <summary>
        /// The optional groupID column that the ranking trainers expects.
        /// </summary>
        public readonly SchemaShape.Column GroupIdColumn;

        [BestFriend]
        private protected TrainerEstimatorBaseWithGroupId(IHost host,
                SchemaShape.Column feature,
                SchemaShape.Column label,
                SchemaShape.Column weight = default,
                SchemaShape.Column groupId = default)
            : base(host, feature, label, weight)
        {
            GroupIdColumn = groupId;
        }

        private protected override RoleMappedData MakeRoles(IDataView data) =>
            new RoleMappedData(data, label: LabelColumn.Name, feature: FeatureColumn.Name, group: GroupIdColumn.Name, weight: WeightColumn.Name);

    }
}
