// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Microsoft.ML.Models
{
    /// <summary>
    /// The confusion matrix shows the predicted values vs the actual values.
    /// Each row of the matrix represents the instances in a predicted class
    /// while each column represents the instances in the actual class.
    /// </summary>
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

        internal static ConfusionMatrix Create(IHostEnvironment env, IDataView confusionMatrix)
        {
            Contracts.AssertValue(env);
            env.AssertValue(confusionMatrix);

            if (!confusionMatrix.Schema.TryGetColumnIndex(MetricKinds.ColumnNames.Count, out int countColumn))
            {
                env.Except($"ConfusionMatrix data view did not contain a {nameof(MetricKinds.ColumnNames.Count)} column.");
            }

            ColumnType type = confusionMatrix.Schema.GetColumnType(countColumn);
            env.Assert(type.IsVector);

            double[,] elements = new double[type.VectorSize, type.VectorSize];

            IRowCursor cursor = confusionMatrix.GetRowCursor(col => col == countColumn);
            ValueGetter<VBuffer<double>> countGetter = cursor.GetGetter<VBuffer<double>>(countColumn);
            VBuffer<double> countValues = default;

            int valuesRowIndex = 0;
            while (cursor.MoveNext())
            {
                countGetter(ref countValues);
                for (int i = 0; i < countValues.Length; i++)
                {
                    elements[valuesRowIndex, i] = countValues.Values[i];
                }

                valuesRowIndex++;
            }

            var slots = default(VBuffer<DvText>);
            confusionMatrix.Schema.GetMetadata(MetadataUtils.Kinds.SlotNames, countColumn, ref slots);
            string[] classNames = new string[slots.Count];
            for (int i = 0; i < slots.Count; i++)
            {
                classNames[i] = slots.Values[i].ToString();
            }

            return new ConfusionMatrix(elements, classNames);
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
