// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;
using Microsoft.ML.Transforms;

[assembly: LoadableClass(typeof(CountTargetEncodingFeaturizer), null, typeof(SignatureLoadModel),
    "Count Target Encoding Featurizer", CountTargetEncodingFeaturizer.RegistrationName)]

namespace Microsoft.ML.Transforms
{
    internal sealed class CountTargetEncodingFeaturizer : ICanSaveModel
    {
        private readonly IHost _host;
        private readonly int _labelBinCount;
        private readonly int _logOddsCount;
        private readonly MultiCountTableBase _countTables;
        public MultiCountTableBase MultiCountTable => _countTables;

        public float[] PriorCoef { get; }
        public float[] LaplaceScale { get; }

        internal const string RegistrationName = "CountTargetEncoder";
        private static VersionInfo GetVersionInfo()
        {
            return new VersionInfo(
                modelSignature: "DRCTFEAT",
                verWrittenCur: 0x00010001, // Initial
                verReadableCur: 0x00010001,
                verWeCanReadBack: 0x00010001,
                loaderSignature: RegistrationName,
                loaderAssemblyName: typeof(CountTargetEncodingFeaturizer).Assembly.FullName);
        }

        public int ColCount => _countTables.ColCount;

        public ReadOnlySpan<int> SlotCount => _countTables.SlotCount;

        public CountTargetEncodingFeaturizer(IHostEnvironment env, float[] priorCoef, float[] laplaceScale, long labelBinCount, MultiCountTableBase countTable)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckParam(labelBinCount > 1, nameof(labelBinCount), "Must be greater than 1");

            _labelBinCount = (int)labelBinCount;
            _logOddsCount = _labelBinCount == 2 ? 1 : _labelBinCount;
            NumFeatures = _labelBinCount + _logOddsCount + 1;

            PriorCoef = priorCoef;

            LaplaceScale = laplaceScale;

            _host.AssertValue(countTable);
            _countTables = countTable;
        }

        public CountTargetEncodingFeaturizer(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(RegistrationName);
            _host.CheckValue(ctx, nameof(ctx));

            ctx.CheckAtModel(GetVersionInfo());

            // *** Binary format ***
            // int: label bin count
            // int: #features
            // float[]: prior coefficient
            // float[]: laplace scale
            // _countTables

            _labelBinCount = ctx.Reader.ReadInt32();
            _host.CheckDecode(_labelBinCount > 1);
            _logOddsCount = _labelBinCount == 2 ? 1 : _labelBinCount;

            NumFeatures = ctx.Reader.ReadInt32();
            _host.CheckDecode(NumFeatures == _labelBinCount + _logOddsCount + 1);

            // initialize count tables
            ctx.LoadModelOrNull<MultiCountTableBase, SignatureLoadModel>(_host, out _countTables, "CountTables");
            _host.AssertValue(_countTables);

            PriorCoef = ctx.Reader.ReadFloatArray(_countTables.ColCount);
            _host.CheckDecode(PriorCoef.All(x => x > 0));

            LaplaceScale = ctx.Reader.ReadFloatArray(_countTables.ColCount);
            _host.CheckDecode(LaplaceScale.All(x => x >= 0));
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.CheckValue(ctx, nameof(ctx));
            ctx.CheckAtModel();
            ctx.SetVersionInfo(GetVersionInfo());

            // *** Binary format ***
            // int: label bin count
            // int: #features
            // Single: prior coefficient
            // Single: laplace scale
            // _countTable

            ctx.Writer.Write(_labelBinCount);
            ctx.Writer.Write(NumFeatures);
            ctx.Writer.WriteSinglesNoCount(PriorCoef);
            ctx.Writer.WriteSinglesNoCount(LaplaceScale);
            ctx.SaveModel(_countTables, "CountTables");
        }

        public int NumFeatures { get; }

        public void GetFeatureNames(string[] classNames, ref VBuffer<ReadOnlyMemory<char>> featureNames)
        {
            if (classNames == null)
            {
                classNames = new string[_labelBinCount];
                for (int i = 0; i < _labelBinCount; i++)
                    classNames[i] = string.Format("Class{0:000}", i);
            }

            Contracts.Check(Utils.Size(classNames) == _labelBinCount, "incorrect class names");

            var editor = VBufferEditor.Create(ref featureNames, NumFeatures);
            for (int i = 0; i < _labelBinCount; i++)
                editor.Values[i] = $"{classNames[i]}_Count".AsMemory();
            for (int i = 0; i < _logOddsCount; i++)
                editor.Values[_labelBinCount + i] = $"{classNames[i]}_LogOdds".AsMemory();
            editor.Values[NumFeatures - 1] = "IsBackoff".AsMemory();
            featureNames = editor.Commit();
        }

        public void GetFeatures(int iCol, int iSlot, Random rand, long key, Span<float> features)
        {
            _host.Assert(features.Length == NumFeatures);

            // get counts from count table in the first _labelBinCount indices.
            var countsBuffer = features.Slice(0, _labelBinCount);
            var countTable = _countTables[iCol, iSlot];
            countTable.GetCounts(key, countsBuffer);

            // check if it's garbage and replace with garbage counts if true
            float sum = 0;
            foreach (var feat in countsBuffer)
                sum += feat;
            bool isGarbage = sum < countTable.GarbageThreshold;
            if (isGarbage)
            {
                int i = 0;
                foreach (var count in countTable.GarbageCounts)
                    countsBuffer[i++] = count;
            }

            sum = AddLaplacianNoisePerLabel(iCol, rand, countsBuffer);

            // add log odds in the next _logOddsCount indices.
            GenerateLogOdds(iCol, countTable, countsBuffer, features.Slice(_labelBinCount, _logOddsCount), sum);
            _host.Assert(FloatUtils.IsFinite(features));

            // Add the last feature: an indicator for isGarbage.
            features[NumFeatures - 1] = isGarbage ? 1 : 0;
        }

        // Assumes features are filled with the respective counts.
        // Adds laplacian noise if set and returns the sum of the counts.
        private float AddLaplacianNoisePerLabel(int iCol, Random rand, Span<float> counts)
        {
            _host.Assert(_labelBinCount == counts.Length);

            float sum = 0;
            for (int ifeat = 0; ifeat < _labelBinCount; ifeat++)
            {
                if (rand != null && LaplaceScale[iCol] > 0)
                    counts[ifeat] += Stats.SampleFromLaplacian(rand, 0, LaplaceScale[iCol]);

                // Clamp to zero when noise is too big and negative.
                if (counts[ifeat] < 0)
                    counts[ifeat] = 0;

                sum += counts[ifeat];
            }

            return sum;
        }

        // Fills _labelBinCount log odds features. One per class, or only one if 2 classes.
        private void GenerateLogOdds(int iCol, ICountTable countTable, Span<float> counts, Span<float> logOdds, float sum)
        {
            _host.Assert(counts.Length == _labelBinCount);
            _host.Assert(logOdds.Length == _logOddsCount);

            for (int i = 0; i < _logOddsCount; i++)
            {
                _host.Assert(counts[i] >= 0);
                if (counts[i] <= 0 && countTable.PriorFrequencies[i] <= 0 || countTable.PriorFrequencies[i] >= 1)
                    logOdds[i] = 0; // guarding against infinite log-odds
                else
                {
                    logOdds[i] = (float)Math.Log(
                        (counts[i] + PriorCoef[iCol] * countTable.PriorFrequencies[i]) /
                        (sum - counts[i] + PriorCoef[iCol] * (1 - countTable.PriorFrequencies[i])));
                }
            }
        }
    }
}
