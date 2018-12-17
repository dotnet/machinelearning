// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Core.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.CommandLine;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.Internallearn;
using Microsoft.ML.Runtime.Internal.Utilities;
using Microsoft.ML.Runtime.Model;
using Microsoft.ML.Runtime.Numeric;
using Microsoft.ML.Runtime.Training;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;

namespace Microsoft.ML.Trainers
{
    public abstract class NearestNeighborBase<TArgs, TTransformer, TModel> : TrainerEstimatorBase<TTransformer, TModel>
      where TTransformer : ISingleFeaturePredictionTransformer<TModel>
      where TModel : IPredictor
      where TArgs : NearestNeighborBase<TArgs, TTransformer, TModel>.ArgumentsBase, new()
    {
        public abstract class ArgumentsBase : LearnerInputBaseWithLabel
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "Number of neighbors to use by default to determine result.", SortOrder = 50)]
            [TGUI(SuggestedSweeps = "0,5,10,20,40")]
            [TlcModule.SweepableDiscreteParam("K", new object[] { 0, 5, 10, 20, 40 })]
            public int K = Defaults.K;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use inverse distance as weight during scoring.", SortOrder = 51)]
            public bool UseDistanceAsWeight = Defaults.UseDistanceAsWeight;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Use Manhattan distance or Eucilidiean.", SortOrder = 52)]
            public bool UseManhattanDistance = Defaults.UseManhattanDistance;

            internal static class Defaults
            {
                /// <value>The number of neighbors.</value>
                internal const int K = 5;
                /// <value>Radius</value>
                internal const float R = 1.0f;
                internal const bool UseDistanceAsWeight = false;
                internal const bool UseManhattanDistance = false;
            }
        }

        private static readonly TrainerInfo _info = new TrainerInfo(normalization: true, caching: false, supportIncrementalTrain: true);

        public override TrainerInfo Info => _info;

        private const string RegisterName = nameof(NearestNeighborBase<TArgs, TTransformer, TModel>);

        protected readonly int K;

        protected readonly bool UseDistanceAsWeight;
        protected readonly bool UseManhattanDistance;

        internal NearestNeighborBase(IHostEnvironment env, string featureColumn, SchemaShape.Column labelColumn,
           int k, Action<TArgs> advancedSettings)
           : this(env, new TArgs
           {
               FeatureColumn = featureColumn,
               LabelColumn = labelColumn.Name,
               K = k
           },
           labelColumn, advancedSettings)
        {
        }

        internal NearestNeighborBase(IHostEnvironment env,
            TArgs args,
            SchemaShape.Column labelColumn,
            Action<TArgs> advancedSettings = null)
            : base(Contracts.CheckRef(env, nameof(env)).Register(RegisterName), TrainerUtils.MakeR4VecFeature(args.FeatureColumn), labelColumn)
        {
            K = args.K;
            UseDistanceAsWeight = args.UseDistanceAsWeight;
            UseManhattanDistance = args.UseManhattanDistance;
        }

    }
    public abstract class NearestNeighborModelParametersBase<TOutput, TLabelType> : PredictorBase<TOutput>
    {
        private const string RegisterName = nameof(NearestNeighborModelParametersBase<TOutput, TLabelType>);

        public readonly int K;

        public readonly bool UseDistanceAsWeight;
        public readonly bool UseManhattanDistance;

        public readonly ImmutableArray<TLabelType> Labels;
        public readonly ImmutableArray<VBuffer<float>> Objects;
        public readonly int FeatureCount;

        internal NearestNeighborModelParametersBase(IHostEnvironment env, int featureCount, IList<VBuffer<float>> objects, IList<TLabelType> labels,
            bool copyIn, int k, bool useDistanceAsWeight, bool useManhattanDistance)
            : base(env, RegisterName)
        {
            Host.CheckValue(objects, nameof(objects));
            Host.CheckValue(labels, nameof(objects));
            Host.Check(objects.Count == labels.Count);
            K = k;
            UseDistanceAsWeight = useDistanceAsWeight;
            UseManhattanDistance = useManhattanDistance;
            FeatureCount = featureCount;

            var objectArray = new VBuffer<float>[objects.Count];
            for (int i = 0; i < objects.Count; i++)
            {
                Host.CheckParam(objects[i].Length == featureCount,
                    nameof(objects), "Inconsistent dimensions found among examples");
                Host.CheckParam(FloatUtils.IsFinite(objects[i].GetValues()),
                    nameof(objects), "Cannot initialize K-NN model parameters with non-finite objects values");
                if (copyIn)
                    objects[i].CopyTo(ref objectArray[i]);
                else
                    objectArray[i] = objects[i];
            }
            Labels = labels.ToImmutableArray();
            Objects = objectArray.ToImmutableArray();
        }
        protected abstract TLabelType ReadLabel(BinaryReader reader);
        protected abstract void WriteLabel(BinaryWriter writer, TLabelType label);

        internal NearestNeighborModelParametersBase(IHostEnvironment env, ModelLoadContext ctx) :
            base(env, RegisterName, ctx)
        {
            // ***Binary format * **
            // int: K, number of neighbors
            // bool: UseDistanceAsWeight
            // bool: UseManhattanDistance
            // int: n, number of objects
            // int: FeatureCount, length of the object vectors

            // for each object, then:
            //     int: count of this object vector (sparse iff count < dimensionality)
            //     int[count]: only present if sparse, in order indices
            //     Float[count]: object vector values
            //     TLabelType: label value
            K = ctx.Reader.ReadInt32();
            Host.CheckDecode(K >= 0);
            UseDistanceAsWeight = ctx.Reader.ReadBoolean();
            UseManhattanDistance = ctx.Reader.ReadBoolean();
            var n = ctx.Reader.ReadInt32();
            Host.CheckDecode(n > 0);
            FeatureCount = ctx.Reader.ReadInt32();
            Host.CheckDecode(FeatureCount > 0);
            var objects = new VBuffer<float>[n];
            var objectsLabels = new TLabelType[n];
            for (int i = 0; i < n; i++)
            {
                int count = ctx.Reader.ReadInt32();
                Host.CheckDecode(0 <= count && count <= FeatureCount);
                var indices = count < FeatureCount ? ctx.Reader.ReadIntArray(count) : null;
                var values = ctx.Reader.ReadFloatArray(count);
                Host.CheckDecode(FloatUtils.IsFinite(values));
                objects[i] = new VBuffer<float>(FeatureCount, count, values, indices);
                objectsLabels[i] = ReadLabel(ctx.Reader);
            }
            Objects = objects.ToImmutableArray();
            Labels = objectsLabels.ToImmutableArray();
        }

        private protected override void SaveCore(ModelSaveContext ctx)
        {
            base.SaveCore(ctx);
            var writer = ctx.Writer;
            // ***Binary format * **
            // int: K, number of neighbors
            // bool: UseDistanceAsWeight
            // bool: UseManhattanDistance
            // int: n, number of objects
            // int: FeatureCount, length of the object vectors

            // for each object, then:
            //     int: count of this object vector (sparse iff count < dimensionality)
            //     int[count]: only present if sparse, in order indices
            //     Float[count]: object vector values
            //     TLabelType: label value

            writer.Write(K);
            writer.Write(UseDistanceAsWeight);
            writer.Write(UseManhattanDistance);
            writer.Write(Objects.Length);
            writer.Write(FeatureCount);
            for (int i = 0; i < Objects.Length; i++)
            {
                Contracts.Assert(Objects[i].Length == FeatureCount);
                var values = Objects[i].GetValues();
                writer.Write(values.Length);
                if (!Objects[i].IsDense)
                    writer.WriteIntsNoCount(Objects[i].GetIndices());
                Contracts.Assert(FloatUtils.IsFinite(values));
                writer.WriteSinglesNoCount(values);
                WriteLabel(writer, Labels[i]);
            }
        }

        private class DistanceAndLabel
        {
            public float Distance { get; }
            public TLabelType Label { get; }
            public DistanceAndLabel(float distance, TLabelType label)
            {
                Distance = distance;
                Label = label;
            }
        }

        protected (IList<float>, IList<TLabelType>) GetScores(in VBuffer<float> src)
        {
            Host.Check(src.Length == FeatureCount, "Invalid number of features passed.");
            var srcValues = src.GetValues();
            var srcIndices = src.GetIndices();
            var weights = new List<float>();
            var labels = new List<TLabelType>();
            var heap = new Heap<DistanceAndLabel>((s1, s2) => s1.Distance < s2.Distance);
            for (int i = 0; i < Objects.Length; i++)
            {
                var dist = UseManhattanDistance ? VectorUtils.L1Distance(src, Objects[i]) : VectorUtils.L2DistSquared(src, Objects[i]);
                heap.Add(new DistanceAndLabel(dist, Labels[i]));
                if (heap.Count > K)
                    heap.Pop();
            }
            float weight = 1.0f / heap.Count;
            while (heap.Count != 0)
            {
                var obj = heap.Pop();
                if (UseDistanceAsWeight)
                {
                    if (obj.Distance == 0)
                        weight = 1;
                    else
                        weight = 1 / obj.Distance;
                }
                weights.Add(weight);
                labels.Add(obj.Label);
            }
            return (weights, labels);
        }

    }
}
