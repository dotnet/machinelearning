// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    internal sealed class DraculaFeaturizer
    {
        public class Options
        {
            [Argument(ArgumentType.AtMostOnce, HelpText = "The coefficient with which to apply the prior smoothing to the features", ShortName = "prior")]
            public float PriorCoefficient = 1;

            [Argument(ArgumentType.AtMostOnce, HelpText = "Laplacian noise diversity/scale-parameter. Suggest keeping it less than 1.", ShortName = "laplace")]
            public float LaplaceScale = 0;
        }

        // these values are initialized at load / creation time
        private readonly IHost _host;
        private readonly int _labelBinCount;
        private readonly int _logOddsCount;
        private readonly float _priorCoef;
        private readonly float _laplaceScale;

        private readonly ICountTable _countTable;
        private readonly float[] _garbageCounts;
        private readonly double[] _priorFrequencies; // prior counts normalized to sum up to 1. Double precision
        private readonly float _garbageThreshold;

        private const string LoaderSignature = "DraculaFeaturizer";

        public DraculaFeaturizer(IHostEnvironment env, Options args, long labelBinCount, ICountTable countTable)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckParam(labelBinCount > 1, nameof(labelBinCount), "Must be greater than 1");

            _labelBinCount = (int)labelBinCount;
            _logOddsCount = _labelBinCount == 2 ? 1 : _labelBinCount;
            NumFeatures = _labelBinCount + _logOddsCount + 1;

            _host.CheckUserArg(args.PriorCoefficient > 0, nameof(args.PriorCoefficient), "Must be greater than zero");
            _priorCoef = args.PriorCoefficient;

            _host.CheckUserArg(args.LaplaceScale >= 0, nameof(args.LaplaceScale), "Must be greater than or equal to zero.");
            _laplaceScale = args.LaplaceScale;

            // initialize arrays
            _garbageCounts = new float[_labelBinCount];
            _priorFrequencies = new double[_labelBinCount];

            // initialize count table and priors
            _host.AssertValue(countTable);
            _countTable = countTable;
            _garbageThreshold = _countTable.GarbageThreshold;
            InitializePriors();
        }

        public DraculaFeaturizer(IHostEnvironment env, ModelLoadContext ctx)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoaderSignature);
            _host.CheckValue(ctx, nameof(ctx));

            // *** Binary format ***
            // int: label bin count
            // int: #features
            // Single: prior coefficient
            // Single: laplace scale
            // _countTable

            _labelBinCount = ctx.Reader.ReadInt32();
            _host.CheckDecode(_labelBinCount > 1);
            _logOddsCount = _labelBinCount == 2 ? 1 : _labelBinCount;

            NumFeatures = ctx.Reader.ReadInt32();
            _host.CheckDecode(NumFeatures == _labelBinCount + _logOddsCount + 1);

            _priorCoef = ctx.Reader.ReadSingle();
            _host.CheckDecode(_priorCoef > 0);

            _laplaceScale = ctx.Reader.ReadSingle();
            _host.CheckDecode(_laplaceScale >= 0);

            // initialize arrays
            _garbageCounts = new float[_labelBinCount];
            _priorFrequencies = new double[_labelBinCount];

            // initialize count table and priors
            ctx.LoadModelOrNull<ICountTable, SignatureLoadModel>(_host, out _countTable, "CountTable");
            _host.AssertValue(_countTable);
            _garbageThreshold = _countTable.GarbageThreshold;
            InitializePriors();
        }

        public void Save(ModelSaveContext ctx)
        {
            Contracts.AssertValue(ctx);

            // *** Binary format ***
            // int: label bin count
            // int: #features
            // Single: prior coefficient
            // Single: laplace scale
            // _countTable

            ctx.Writer.Write(_labelBinCount);
            ctx.Writer.Write(NumFeatures);
            ctx.Writer.Write(_priorCoef);
            ctx.Writer.Write(_laplaceScale);
            ctx.SaveModel(_countTable, "CountTable");
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

            _host.Check(Utils.Size(classNames) == _labelBinCount, "incorrect class names");

            var editor = VBufferEditor.Create(ref featureNames, NumFeatures);
            for (int i = 0; i < _labelBinCount; i++)
                editor.Values[i] = $"{classNames[i]}_Count".AsMemory();
            for (int i = 0; i < _logOddsCount; i++)
                editor.Values[_labelBinCount + i] = $"{classNames[i]}_LogOdds".AsMemory();
            editor.Values[NumFeatures-1] = "IsBackoff".AsMemory();
            featureNames = editor.Commit();
        }

        private void InitializePriors()
        {
            var priorCounts = new float[_labelBinCount];

            _countTable.GetPriors(priorCounts, _garbageCounts);

            _host.Check(priorCounts.All(x => (x >= 0)));
            var priorSum = priorCounts.Sum();
            if (priorSum > 0)
            {
                for (int i = 0; i < _labelBinCount; i++)
                    _priorFrequencies[i] = priorCounts[i] / priorSum;

                return;
            }

            // if there is no prior computed, defer to 1/N
            var d = 1.0 / _labelBinCount;
            for (int i = 0; i < _labelBinCount; i++)
                _priorFrequencies[i] = d;
        }

        public void GetFeatures(Random rand, long key, Span<float> features)
        {
            _host.AssertValue(_countTable);
            _host.Assert(features.Length == NumFeatures);

            // get counts from count table in the first _labelBinCount indices.
            var countsBuffer = features.Slice(0, _labelBinCount);
            //var countsBuffer = _countsBuffers.Get();
            _countTable.GetCounts(key, countsBuffer);

            // check if it's garbage and replace with garbage counts if true
            //var sum = countsBuffer.Sum();
            float sum = 0;
            foreach (var feat in countsBuffer)
                sum += feat;
            bool isGarbage = sum <= _garbageThreshold;
            if (isGarbage)
                _garbageCounts.CopyTo(countsBuffer);
            //Array.Copy(_garbageCounts, countsBuffer, _labelBinCount);

            //sum = AddLaplacianNoisePerLabel(countsBuffer);
            sum = AddLaplacianNoisePerLabel(rand, countsBuffer);

            // add log odds in the next _logOddsCount indices.
            GenerateLogOdds(countsBuffer, features.Slice(_labelBinCount, _logOddsCount), sum);
            AssertValidOutput(features);

            //// populate raw counts and garbage flag
            //Array.Copy(countsBuffer, 0, features, startIdx, _labelBinCount);

            // Add the last feature: an indicator for isGarbage.
            features[NumFeatures - 1] = isGarbage ? 1 : 0;
        }

        // Assumes features are filled with the respective counts.
        // Adds laplacian noise if set and returns the sum of the counts.
        private float AddLaplacianNoisePerLabel(Random rand, Span<float> counts)
        {
            _host.Assert(_labelBinCount == counts.Length);

            float sum = 0;
            for (int ifeat = 0; ifeat < _labelBinCount; ifeat++)
            {
                if (_laplaceScale > 0)
                    counts[ifeat] += Stats.SampleFromLaplacian(rand, 0, _laplaceScale);

                // Clamp to zero when noise is too big and negative.
                if (counts[ifeat] < 0)
                    counts[ifeat] = 0;

                sum += counts[ifeat];
            }

            return sum;
        }

        // Fills _labelBinCount log odds features. One per class, or only one if 2 classes.
        private void GenerateLogOdds(Span<float> counts, Span<float> logOdds, Single sum)
        {
            _host.Assert(counts.Length == _labelBinCount);
            _host.Assert(logOdds.Length == _logOddsCount);

            for (int i = 0; i < _logOddsCount; i++)
            {
                _host.Assert(counts[i] >= 0);
                if (counts[i] <= 0 && _priorFrequencies[i] <= 0 || _priorFrequencies[i] >= 1)
                    logOdds[i] = 0; // guarding against infinite log-odds
                else
                {
                    logOdds[i] = (float)Math.Log(
                        (counts[i] + _priorCoef * _priorFrequencies[i]) /
                        (sum - counts[i] + _priorCoef * (1 - _priorFrequencies[i])));
                }
            }
        }

        [Conditional("DEBUG")]
        private void AssertValidOutput(Span<float> features)
        {
            foreach (var feature in features)
                _host.Assert(FloatUtils.IsFinite(feature));
        }
    }
}