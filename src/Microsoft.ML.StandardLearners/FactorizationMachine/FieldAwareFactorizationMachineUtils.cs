// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Internal.CpuMath;
using Microsoft.ML.Runtime.Internal.Utilities;

namespace Microsoft.ML.Runtime.FactorizationMachine
{
    internal sealed class FieldAwareFactorizationMachineUtils
    {
        internal static int GetAlignedVectorLength(int length)
        {
            int res = length % 4;
            if (res == 0)
                return length;
            else
                return length + (4 - res);
        }

        internal static bool LoadOneExampleIntoBuffer(ValueGetter<VBuffer<float>>[] getters, VBuffer<float> featureBuffer, bool norm, ref int count,
            int[] fieldIndexBuffer, int[] featureIndexBuffer, float[] featureValueBuffer)
        {
            count = 0;
            float featureNorm = 0;
            int bias = 0;
            float annihilation = 0;
            for (int f = 0; f < getters.Length; f++)
            {
                getters[f](ref featureBuffer);
                foreach (var pair in featureBuffer.Items())
                {
                    fieldIndexBuffer[count] = f;
                    featureIndexBuffer[count] = bias + pair.Key;
                    featureValueBuffer[count] = pair.Value;
                    featureNorm += pair.Value * pair.Value;
                    annihilation += pair.Value - pair.Value;
                    count++;
                }
                bias += featureBuffer.Length;
            }
            featureNorm = MathUtils.Sqrt(featureNorm);
            if (norm)
            {
                for (int i = 0; i < count; i++)
                    featureValueBuffer[i] /= featureNorm;
            }
            return FloatUtils.IsFinite(annihilation);
        }
    }

    internal sealed class FieldAwareFactorizationMachineScalarRowMapper : ISchemaBoundRowMapper
    {
        private readonly FieldAwareFactorizationMachinePredictor _pred;

        public RoleMappedSchema InputSchema { get; }

        public ISchema OutputSchema { get; }

        public ISchemaBindableMapper Bindable => _pred;

        private readonly ColumnInfo[] _columns;
        private readonly List<int> _inputColumnIndexes;
        private readonly IHostEnvironment _env;

        public FieldAwareFactorizationMachineScalarRowMapper(IHostEnvironment env, RoleMappedSchema schema,
            ISchema outputSchema, FieldAwareFactorizationMachinePredictor pred)
        {
            Contracts.AssertValue(env);
            Contracts.AssertValue(schema);
            Contracts.CheckParam(outputSchema.ColumnCount == 2, nameof(outputSchema));
            Contracts.CheckParam(outputSchema.GetColumnType(0).IsNumber, nameof(outputSchema));
            Contracts.CheckParam(outputSchema.GetColumnType(1).IsNumber, nameof(outputSchema));
            Contracts.AssertValue(pred);

            _env = env;
            _columns = schema.GetColumns(RoleMappedSchema.ColumnRole.Feature).ToArray();
            _pred = pred;

            var inputFeatureColumns = _columns.Select(c => new KeyValuePair<RoleMappedSchema.ColumnRole, string>(RoleMappedSchema.ColumnRole.Feature, c.Name)).ToList();
            InputSchema = new RoleMappedSchema(schema.Schema, inputFeatureColumns);
            OutputSchema = outputSchema;

            _inputColumnIndexes = new List<int>();
            foreach (var kvp in inputFeatureColumns)
            {
                if (schema.Schema.TryGetColumnIndex(kvp.Value, out int index))
                    _inputColumnIndexes.Add(index);
            }
        }

        public IRow GetOutputRow(IRow input, Func<int, bool> predicate, out Action action)
        {
            var latentSum = new AlignedArray(_pred.FieldCount * _pred.FieldCount * _pred.LatentDimAligned, 16);
            var featureBuffer = new VBuffer<float>();
            var featureFieldBuffer = new int[_pred.FeatureCount];
            var featureIndexBuffer = new int[_pred.FeatureCount];
            var featureValueBuffer = new float[_pred.FeatureCount];
            var inputGetters = new ValueGetter<VBuffer<float>>[_pred.FieldCount];
            for (int f = 0; f < _pred.FieldCount; f++)
                inputGetters[f] = input.GetGetter<VBuffer<float>>(_inputColumnIndexes[f]);

            action = null;
            var getters = new Delegate[2];
            if (predicate(0))
            {
                ValueGetter<float> responseGetter = (ref float value) =>
                {
                    value = _pred.CalculateResponse(inputGetters, featureBuffer, featureFieldBuffer, featureIndexBuffer, featureValueBuffer, latentSum);
                };
                getters[0] = responseGetter;
            }
            if (predicate(1))
            {
                ValueGetter<float> probGetter = (ref float value) =>
                {
                    value = _pred.CalculateResponse(inputGetters, featureBuffer, featureFieldBuffer, featureIndexBuffer, featureValueBuffer, latentSum);
                    value = MathUtils.SigmoidSlow(value);
                };
                getters[1] = probGetter;
            }

            return new SimpleRow(OutputSchema, input, getters);
        }

        public Func<int, bool> GetDependencies(Func<int, bool> predicate)
        {
            if (Enumerable.Range(0, OutputSchema.ColumnCount).Any(predicate))
                return index => _inputColumnIndexes.Any(c => c == index);
            else
                return index => false;
        }

        public IEnumerable<KeyValuePair<RoleMappedSchema.ColumnRole, string>> GetInputColumnRoles()
        {
            return InputSchema.GetColumnRoles().Select(kvp => new KeyValuePair<RoleMappedSchema.ColumnRole, string>(kvp.Key, kvp.Value.Name));
        }
    }
}
