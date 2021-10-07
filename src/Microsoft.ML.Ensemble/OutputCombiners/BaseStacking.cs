// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Internallearn;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Trainers.Ensemble
{
    internal abstract class BaseStacking<TOutput> : IStackingTrainer<TOutput>, ICanSaveModel
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "vp", SortOrder = 50,
                HelpText = "The proportion of instances to be selected to test the individual base learner. If it is 0, it uses training set")]
            [TGUI(Label = "Validation Dataset Proportion")]
            public Single ValidationDatasetProportion = 0.3f;

            internal abstract IComponentFactory<ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<TOutput>>, IPredictorProducing<TOutput>>> GetPredictorFactory();
        }

        private protected readonly IComponentFactory<ITrainerEstimator<ISingleFeaturePredictionTransformer<IPredictorProducing<TOutput>>, IPredictorProducing<TOutput>>> BasePredictorType;
        private protected readonly IHost Host;
        private protected IPredictorProducing<TOutput> Meta;

        public Single ValidationDatasetProportion { get; }

        private protected BaseStacking(IHostEnvironment env, string name, ArgumentsBase args)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.AssertValue(args, "args");
            Host.CheckUserArg(0 <= args.ValidationDatasetProportion && args.ValidationDatasetProportion < 1,
                    nameof(args.ValidationDatasetProportion),
                    "The validation proportion for stacking should be greater than or equal to 0 and less than 1");

            ValidationDatasetProportion = args.ValidationDatasetProportion;
            BasePredictorType = args.GetPredictorFactory();
            Host.CheckValue(BasePredictorType, nameof(BasePredictorType));
        }

        private protected BaseStacking(IHostEnvironment env, string name, ModelLoadContext ctx)
        {
            Contracts.AssertValue(env);
            env.AssertNonWhiteSpace(name);
            Host = env.Register(name);
            Host.AssertValue(ctx);

            // *** Binary format ***
            // int: sizeof(Single)
            // Float: _validationDatasetProportion
            int cbFloat = ctx.Reader.ReadInt32();
            env.CheckDecode(cbFloat == sizeof(Single));
            ValidationDatasetProportion = ctx.Reader.ReadFloat();
            env.CheckDecode(0 <= ValidationDatasetProportion && ValidationDatasetProportion < 1);

            ctx.LoadModel<IPredictorProducing<TOutput>, SignatureLoadModel>(env, out Meta, "MetaPredictor");
            CheckMeta();
        }

        void ICanSaveModel.Save(ModelSaveContext ctx)
        {
            Host.Check(Meta != null, "Can't save an untrained Stacking combiner");
            Host.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            SaveCore(ctx);
        }

        protected virtual void SaveCore(ModelSaveContext ctx)
        {
            Host.Assert(Meta != null);

            // *** Binary format ***
            // int: sizeof(Single)
            // Float: _validationDatasetProportion
            ctx.Writer.Write(sizeof(Single));
            ctx.Writer.Write(ValidationDatasetProportion);

            ctx.SaveModel(Meta, "MetaPredictor");
        }

        public Combiner<TOutput> GetCombiner()
        {
            Contracts.Check(Meta != null, "Training of stacking combiner not complete");

            // Subtle point: We shouldn't get the ValueMapper delegate and cache it in a field
            // since generally ValueMappers cannot be assumed to be thread safe - they often
            // capture buffers needed for efficient operation.
            var mapper = (IValueMapper)Meta;
            var map = mapper.GetMapper<VBuffer<Single>, TOutput>();

            var feat = default(VBuffer<Single>);
            Combiner<TOutput> res =
                (ref TOutput dst, TOutput[] src, Single[] weights) =>
                {
                    FillFeatureBuffer(src, ref feat);
                    map(in feat, ref dst);
                };
            return res;
        }

        protected abstract void FillFeatureBuffer(TOutput[] src, ref VBuffer<Single> dst);

        private void CheckMeta()
        {
            Contracts.Assert(Meta != null);

            var ivm = Meta as IValueMapper;
            Contracts.Check(ivm != null, "Stacking predictor doesn't implement the expected interface");
            if (!(ivm.InputType is VectorDataViewType vectorType) || vectorType.ItemType != NumberDataViewType.Single)
                throw Contracts.Except("Stacking predictor input type is unsupported: {0}", ivm.InputType);
            if (ivm.OutputType.RawType != typeof(TOutput))
                throw Contracts.Except("Stacking predictor output type is unsupported: {0}", ivm.OutputType);
        }

        public void Train(List<FeatureSubsetModel<TOutput>> models, RoleMappedData data, IHostEnvironment env)
        {
            Contracts.CheckValue(env, nameof(env));
            var host = env.Register(Stacking.LoadName);
            host.CheckValue(models, nameof(models));
            host.CheckValue(data, nameof(data));

            using (var ch = host.Start("Training stacked model"))
            {
                ch.Check(Meta == null, "Train called multiple times");
                ch.Check(BasePredictorType != null);

                var maps = new ValueMapper<VBuffer<Single>, TOutput>[models.Count];
                for (int i = 0; i < maps.Length; i++)
                {
                    Contracts.Assert(models[i].Predictor is IValueMapper);
                    var m = (IValueMapper)models[i].Predictor;
                    maps[i] = m.GetMapper<VBuffer<Single>, TOutput>();
                }

                var view = CreateDataView(host, ch, data, maps, models);
                var trainer = BasePredictorType.CreateComponent(host);
                if (trainer.Info.NeedNormalization)
                    ch.Warning("The trainer specified for stacking wants normalization, but we do not currently allow this.");
                Meta = trainer.Fit(view).Model;
                CheckMeta();
            }
        }

        private IDataView CreateDataView(IHostEnvironment env, IChannel ch, RoleMappedData data, ValueMapper<VBuffer<Single>,
            TOutput>[] maps, List<FeatureSubsetModel<TOutput>> models)
        {
            switch (data.Schema.Label.Value.Type.GetRawKind())
            {
                case InternalDataKind.BL:
                    return CreateDataView<bool>(env, ch, data, maps, models, x => x > 0);
                case InternalDataKind.R4:
                    return CreateDataView<float>(env, ch, data, maps, models, x => x);
                case InternalDataKind.U4:
                    ch.Check(data.Schema.Label.Value.Type is KeyDataViewType);
                    return CreateDataView(env, ch, data, maps, models, x => float.IsNaN(x) ? 0 : (uint)(x + 1));
                default:
                    throw ch.Except("Unsupported label type");
            }
        }

        private IDataView CreateDataView<T>(IHostEnvironment env, IChannel ch, RoleMappedData data, ValueMapper<VBuffer<Single>, TOutput>[] maps,
            List<FeatureSubsetModel<TOutput>> models, Func<float, T> labelConvert)
        {
            // REVIEW: Should implement this better....
            var labels = new T[100];
            var features = new VBuffer<Single>[100];
            int count = 0;
            // REVIEW: Should this include bad values or filter them?
            using (var cursor = new FloatLabelCursor(data, CursOpt.AllFeatures | CursOpt.AllLabels))
            {
                TOutput[] predictions = new TOutput[maps.Length];
                var vBuffers = new VBuffer<Single>[maps.Length];
                while (cursor.MoveNext())
                {
                    Parallel.For(0, maps.Length, i =>
                    {
                        var model = models[i];
                        if (model.SelectedFeatures != null)
                        {
                            EnsembleUtils.SelectFeatures(in cursor.Features, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                            maps[i](in vBuffers[i], ref predictions[i]);
                        }
                        else
                            maps[i](in cursor.Features, ref predictions[i]);
                    });

                    Utils.EnsureSize(ref labels, count + 1);
                    Utils.EnsureSize(ref features, count + 1);
                    labels[count] = labelConvert(cursor.Label);
                    FillFeatureBuffer(predictions, ref features[count]);
                    count++;
                }
            }

            ch.Info("The number of instances used for stacking trainer is {0}", count);

            var bldr = new ArrayDataViewBuilder(env);
            Array.Resize(ref labels, count);
            Array.Resize(ref features, count);
            bldr.AddColumn(DefaultColumnNames.Label, data.Schema.Label.Value.Type as PrimitiveDataViewType, labels);
            bldr.AddColumn(DefaultColumnNames.Features, NumberDataViewType.Single, features);

            return bldr.GetDataView();
        }
    }
}
