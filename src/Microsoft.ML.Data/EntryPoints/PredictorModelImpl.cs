// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Model;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.EntryPoints
{
    /// <summary>
    /// This class encapsulates the predictor and a preceding transform model, as the concrete and hidden
    /// implementation of <see cref="PredictorModel"/>.
    /// </summary>
    [BestFriend]
    internal sealed class PredictorModelImpl : PredictorModel
    {
        private readonly KeyValuePair<RoleMappedSchema.ColumnRole, string>[] _roleMappings;

        internal override TransformModel TransformModel { get; }

        internal override IPredictor Predictor { get; }

        [BestFriend]
        internal PredictorModelImpl(IHostEnvironment env, RoleMappedData trainingData, IDataView startingData, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(trainingData, nameof(trainingData));
            env.CheckValue(predictor, nameof(predictor));

            TransformModel = new TransformModelImpl(env, trainingData.Data, startingData);
            _roleMappings = trainingData.Schema.GetColumnRoleNames().ToArray();
            Predictor = predictor;
        }

        [BestFriend]
        internal PredictorModelImpl(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));
            using (var ch = env.Start("Loading predictor model"))
            {
                // REVIEW: address the asymmetry in the way we're loading and saving the model.
                TransformModel = new TransformModelImpl(env, stream);

                var roles = ModelFileUtils.LoadRoleMappingsOrNull(env, stream);
                env.CheckDecode(roles != null, "Predictor model must contain role mappings");
                _roleMappings = roles.ToArray();

                Predictor = ModelFileUtils.LoadPredictorOrNull(env, stream);
                env.CheckDecode(Predictor != null, "Predictor model must contain a predictor");
            }
        }

        private PredictorModelImpl(TransformModel transformModel, IPredictor predictor, KeyValuePair<RoleMappedSchema.ColumnRole, string>[] roleMappings)
        {
            Contracts.AssertValue(transformModel);
            Contracts.AssertValue(predictor);
            Contracts.AssertValue(roleMappings);
            TransformModel = transformModel;
            Predictor = predictor;
            _roleMappings = roleMappings;
        }

        internal override void Save(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));
            using (var ch = env.Start("Saving predictor model"))
            {
                // REVIEW: address the asymmetry in the way we're loading and saving the model.
                // Effectively, we have methods to load the transform model from a model.zip, but don't have
                // methods to compose the model.zip out of transform model, predictor and role mappings
                // (we use the TrainUtils.SaveModel that does all three).

                // Create the chain of transforms for saving.
                IDataView data = new EmptyDataView(env, TransformModel.InputSchema);
                data = TransformModel.Apply(env, data);
                var roleMappedData = new RoleMappedData(data, _roleMappings, opt: true);

                TrainUtils.SaveModel(env, ch, stream, Predictor, roleMappedData);
            }
        }

        internal override PredictorModel Apply(IHostEnvironment env, TransformModel transformModel)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformModel, nameof(transformModel));
            TransformModel newTransformModel = TransformModel.Apply(env, transformModel);
            Contracts.AssertValue(newTransformModel);
            return new PredictorModelImpl(newTransformModel, Predictor, _roleMappings);
        }

        internal override void PrepareData(IHostEnvironment env, IDataView input, out RoleMappedData roleMappedData, out IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            input = TransformModel.Apply(env, input);
            roleMappedData = new RoleMappedData(input, _roleMappings, opt: true);
            predictor = Predictor;
        }

        internal override string[] GetLabelInfo(IHostEnvironment env, out DataViewType labelType)
        {
            Contracts.CheckValue(env, nameof(env));
            var predictor = Predictor;
            var calibrated = predictor as IWeaklyTypedCalibratedModelParameters;
            while (calibrated != null)
            {
                predictor = calibrated.WeaklyTypedSubModel;
                calibrated = predictor as IWeaklyTypedCalibratedModelParameters;
            }
            var canGetTrainingLabelNames = predictor as ICanGetTrainingLabelNames;
            if (canGetTrainingLabelNames != null)
                return canGetTrainingLabelNames.GetLabelNamesOrNull(out labelType);

            var trainRms = GetTrainingSchema(env);
            labelType = null;
            if (trainRms.Label != null)
            {
                labelType = trainRms.Label.Value.Type;
                if (trainRms.Label.Value.HasKeyValues())
                {
                    VBuffer<ReadOnlyMemory<char>> keyValues = default;
                    trainRms.Label.Value.GetKeyValues(ref keyValues);
                    return keyValues.DenseValues().Select(v => v.ToString()).ToArray();
                }
            }
            return null;
        }

        internal override RoleMappedSchema GetTrainingSchema(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            var predInput = TransformModel.Apply(env, new EmptyDataView(env, TransformModel.InputSchema));
            var trainRms = new RoleMappedSchema(predInput.Schema, _roleMappings, opt: true);
            return trainRms;
        }
    }
}
