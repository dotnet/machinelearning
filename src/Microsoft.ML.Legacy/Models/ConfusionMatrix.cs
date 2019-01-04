﻿// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using Microsoft.ML.Data;

namespace Microsoft.ML.Legacy.Models
{
    /// <summary>
    /// The confusion matrix shows the predicted values vs the actual values.
    /// Each row of the matrix represents the instances in a predicted class
    /// while each column represents the instances in the actual class.
    /// </summary>
    [Obsolete]
    public sealed class ConfusionMatrix
    {
        private readonly double[,] _elements;
        private readonly string[] _classNames;
        private readonly Lazy<Dictionary<string, int>> _classNameIndex;

        private ConfusionMatrix(double[,] elements, string[] classNames)
        {
            Contracts.AssertValue(elements, nameof(elements));
            Contracts.Assert(elements.GetLength(0) == elements.GetLength(1), $"{nameof(elements)} must be a square matrix.");
            Contracts.AssertValue(classNames, nameof(classNames));
            Contracts.Assert(classNames.Length == elements.GetLength(0));

            _elements = elements;
            _classNames = classNames;
            _classNameIndex = new Lazy<Dictionary<string, int>>(() =>
            {
                Dictionary<string, int> result = new Dictionary<string, int>();
                for (int i = 0; i < _classNames.Length; i++)
                {
                    result[_classNames[i]] = i;
                }
                return result;
            });
        }

        internal static List<ConfusionMatrix> Create(IHostEnvironment env, IDataView confusionMatrix)
        {
            Contracts.AssertValue(env);
            env.AssertValue(confusionMatrix);

            if (!confusionMatrix.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.Count, out int countColumn))
            {
                throw env.Except($"ConfusionMatrix data view did not contain a {nameof(MetricKinds.ColumnNames.Count)} column.");
            }

            RowCursor cursor = confusionMatrix.GetRowCursor(col => col == countColumn);
            var slots = default(VBuffer<ReadOnlyMemory<char>>);
            confusionMatrix.Schema[countColumn].Metadata.GetValue(MetadataUtils.Kinds.SlotNames, ref slots);
            var slotsValues = slots.GetValues();
            string[] classNames = new string[slotsValues.Length];
            for (int i = 0; i < slotsValues.Length; i++)
            {
                classNames[i] = slotsValues[i].ToString();
            }

            ColumnType type = confusionMatrix.Schema[countColumn].Type;
            env.Assert(type.IsVector);
            ValueGetter<VBuffer<double>> countGetter = cursor.GetGetter<VBuffer<double>>(countColumn);
            VBuffer<double> countValues = default;
            List<ConfusionMatrix> confusionMatrices = new List<ConfusionMatrix>();

            int valuesRowIndex = 0;
            double[,] elements = null;
            while (cursor.MoveNext())
            {
                if(valuesRowIndex == 0)
                    elements = new double[type.VectorSize, type.VectorSize];

                countGetter(ref countValues);
                ReadOnlySpan<double> values = countValues.GetValues();
                for (int i = 0; i < values.Length; i++)
                {
                    elements[valuesRowIndex, i] = values[i];
                }

                valuesRowIndex++;

                if(valuesRowIndex == type.VectorSize)
                {
                    valuesRowIndex = 0;
                    confusionMatrices.Add(new ConfusionMatrix(elements, classNames));
                }
            }

            return confusionMatrices;
        }

        /// <summary>
        /// Gets the number of rows or columns in the matrix.
        /// </summary>
        public int Order => _classNames.Length;

        /// <summary>
        /// Gets the class names of the confusion matrix in the same
        /// order as the rows/columns.
        /// </summary>
        public IReadOnlyList<string> ClassNames => _classNames;

        /// <summary>
        /// Obtains the value at the specified indices.
        /// </summary>
        /// <param name="x">
        /// The row index to retrieve.
        /// </param>
        /// <param name="y">
        /// The column index to retrieve.
        /// </param>
        public double this[int x, int y]
        {
            get
            {
                if (x < 0 || x >= Order)
                    throw new ArgumentOutOfRangeException(nameof(x));
                if (y < 0 || y >= Order)
                    throw new ArgumentOutOfRangeException(nameof(y));

                return _elements[x, y];
            }
        }

        /// <summary>
        /// Obtains the value for the specified class names.
        /// </summary>
        /// <param name="x">
        /// The name of the class for which row to retrieve.
        /// </param>
        /// <param name="y">
        /// The name of the class for which column to retrieve.
        /// </param>
        public double this[string x, string y]
        {
            get
            {
                if (!_classNameIndex.Value.TryGetValue(x, out int xIndex))
                    throw new ArgumentOutOfRangeException(nameof(x));

                if (!_classNameIndex.Value.TryGetValue(y, out int yIndex))
                    throw new ArgumentOutOfRangeException(nameof(y));

                return this[xIndex, yIndex];
            }
        }
    }
}
