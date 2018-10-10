// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Training;

namespace Microsoft.ML.Runtime.Ensemble.OutputCombiners
{
    using ColumnRole = RoleMappedSchema.ColumnRole;
    public abstract class BaseStacking<TOutput> : IStackingTrainer<TOutput>
    {
        public abstract class ArgumentsBase
        {
            [Argument(ArgumentType.AtMostOnce, ShortName = "vp", SortOrder = 50,
                HelpText = "The proportion of instances to be selected to test the individual base learner. If it is 0, it uses training set")]
            [TGUI(Label = "Validation Dataset Proportion")]
            public Single ValidationDatasetProportion = 0.3f;

            internal abstract IComponentFactory<ITrainer<IPredictorProducing<TOutput>>> GetPredictorFactory();
        }

        protected readonly IComponentFactory<ITrainer<IPredictorProducing<TOutput>>> BasePredictorType;
        protected readonly IHost Host;
        protected IPredictorProducing<TOutput> Meta;

        public Single ValidationDatasetProportion { get; }

        internal BaseStacking(IHostEnvironment env, string name, ArgumentsBase args)
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

        internal BaseStacking(IHostEnvironment env, string name, ModelLoadContext ctx)
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

        public void Save(ModelSaveContext ctx)
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
                    map(ref feat, ref dst);
                };
            return res;
        }

        protected abstract void FillFeatureBuffer(TOutput[] src, ref VBuffer<Single> dst);

        private void CheckMeta()
        {
            Contracts.Assert(Meta != null);

            var ivm = Meta as IValueMapper;
            Contracts.Check(ivm != null, "Stacking predictor doesn't implement the expected interface");
            if (!ivm.InputType.IsVector || ivm.InputType.ItemType != NumberType.Float)
                throw Contracts.Except("Stacking predictor input type is unsupported: {0}", ivm.InputType);
            if (ivm.OutputType.RawType != typeof(TOutput))
                throw Contracts.Except("Stacking predictor output type is unsupported: {0}", ivm.OutputType);
        }

        public void Train(List<FeatureSubsetModel<IPredictorProducing<TOutput>>> models, RoleMappedData data, IHostEnvironment env)
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

                // REVIEW: Should implement this better....
                var labels = new Single[100];
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
                                EnsembleUtils.SelectFeatures(ref cursor.Features, model.SelectedFeatures, model.Cardinality, ref vBuffers[i]);
                                maps[i](ref vBuffers[i], ref predictions[i]);
                            }
                            else
                                maps[i](ref cursor.Features, ref predictions[i]);
                        });

                        Utils.EnsureSize(ref labels, count + 1);
                        Utils.EnsureSize(ref features, count + 1);
                        labels[count] = cursor.Label;
                        FillFeatureBuffer(predictions, ref features[count]);
                        count++;
                    }
                }

                ch.Info("The number of instances used for stacking trainer is {0}", count);

                var bldr = new ArrayDataViewBuilder(host);
                Array.Resize(ref labels, count);
                Array.Resize(ref features, count);
                bldr.AddColumn(DefaultColumnNames.Label, NumberType.Float, labels);
                bldr.AddColumn(DefaultColumnNames.Features, NumberType.Float, features);

                var view = bldr.GetDataView();
                var rmd = new RoleMappedData(view, DefaultColumnNames.Label, DefaultColumnNames.Features);

                var trainer = BasePredictorType.CreateComponent(host);
                if (trainer.Info.NeedNormalization)
                    ch.Warning("The trainer specified for stacking wants normalization, but we do not currently allow this.");
                Meta = trainer.Train(rmd);
                CheckMeta();
            }
        }
    }
}
