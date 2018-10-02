// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.Calibration;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Model;

namespace Microsoft.ML.Runtime.EntryPoints
{
    /// <summary>
    /// This class encapsulates the predictor and a preceding transform model.
    /// </summary>
    public sealed class PredictorModel : IPredictorModel
    {
        private readonly IPredictor _predictor;
        private readonly ITransformModel _transformModel;
        private readonly KeyValuePair<RoleMappedSchema.ColumnRole, string>[] _roleMappings;

        public ITransformModel TransformModel { get { return _transformModel; } }

        public IPredictor Predictor { get { return _predictor; } }

        public PredictorModel(IHostEnvironment env, RoleMappedData trainingData, IDataView startingData, IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(trainingData, nameof(trainingData));
            env.CheckValue(predictor, nameof(predictor));

            _transformModel = new TransformModel(env, trainingData.Data, startingData);
            _roleMappings = trainingData.Schema.GetColumnRoleNames().ToArray();
            _predictor = predictor;
        }

        public PredictorModel(IHostEnvironment env, Stream stream)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(stream, nameof(stream));
            using (var ch = env.Start("Loading predictor model"))
            {
                // REVIEW: address the asymmetry in the way we're loading and saving the model.
                _transformModel = new TransformModel(env, stream);

                var roles = ModelFileUtils.LoadRoleMappingsOrNull(env, stream);
                env.CheckDecode(roles != null, "Predictor model must contain role mappings");
                _roleMappings = roles.ToArray();

                _predictor = ModelFileUtils.LoadPredictorOrNull(env, stream);
                env.CheckDecode(_predictor != null, "Predictor model must contain a predictor");

                ch.Done();
            }
        }

        private PredictorModel(ITransformModel transformModel, IPredictor predictor, KeyValuePair<RoleMappedSchema.ColumnRole, string>[] roleMappings)
        {
            Contracts.AssertValue(transformModel);
            Contracts.AssertValue(predictor);
            Contracts.AssertValue(roleMappings);
            _transformModel = transformModel;
            _predictor = predictor;
            _roleMappings = roleMappings;
        }

        public void Save(IHostEnvironment env, Stream stream)
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
                IDataView data = new EmptyDataView(env, _transformModel.InputSchema);
                data = _transformModel.Apply(env, data);
                var roleMappedData = new RoleMappedData(data, _roleMappings, opt: true);

                TrainUtils.SaveModel(env, ch, stream, _predictor, roleMappedData);
                ch.Done();
            }
        }

        public IPredictorModel Apply(IHostEnvironment env, ITransformModel transformModel)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(transformModel, nameof(transformModel));
            ITransformModel newTransformModel = _transformModel.Apply(env, transformModel);
            Contracts.AssertValue(newTransformModel);
            return new PredictorModel(newTransformModel, _predictor, _roleMappings);
        }

        public void PrepareData(IHostEnvironment env, IDataView input, out RoleMappedData roleMappedData, out IPredictor predictor)
        {
            Contracts.CheckValue(env, nameof(env));
            env.CheckValue(input, nameof(input));

            input = _transformModel.Apply(env, input);
            roleMappedData = new RoleMappedData(input, _roleMappings, opt: true);
            predictor = _predictor;
        }

        public string[] GetLabelInfo(IHostEnvironment env, out ColumnType labelType)
        {
            Contracts.CheckValue(env, nameof(env));
            var predictor = _predictor;
            var calibrated = predictor as CalibratedPredictorBase;
            while (calibrated != null)
            {
                predictor = calibrated.SubPredictor;
                calibrated = predictor as CalibratedPredictorBase;
            }
            var canGetTrainingLabelNames = predictor as ICanGetTrainingLabelNames;
            if (canGetTrainingLabelNames != null)
                return canGetTrainingLabelNames.GetLabelNamesOrNull(out labelType);

            var trainRms = GetTrainingSchema(env);
            labelType = null;
            if (trainRms.Label != null)
            {
                labelType = trainRms.Label.Type;
                if (labelType.IsKey &&
                    trainRms.Schema.HasKeyNames(trainRms.Label.Index, labelType.KeyCount))
                {
                    VBuffer<ReadOnlyMemory<char>> keyValues = default;
                    trainRms.Schema.GetMetadata(MetadataUtils.Kinds.KeyValues, trainRms.Label.Index,
                        ref keyValues);
                    return keyValues.DenseValues().Select(v => v.ToString()).ToArray();
                }
            }
            return null;
        }

        public RoleMappedSchema GetTrainingSchema(IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            var predInput = _transformModel.Apply(env, new EmptyDataView(env, _transformModel.InputSchema));
            var trainRms = new RoleMappedSchema(predInput.Schema, _roleMappings, opt: true);
            return trainRms;
        }
    }
}
