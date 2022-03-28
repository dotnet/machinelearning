// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.SearchSpace;
using Microsoft.ML.AutoML.CodeGen;

namespace Microsoft.ML.AutoML
{
    internal class Estimator
    {
        protected Estimator()
        {
            this.Parameter = Parameter.CreateNestedParameter();
            this.EstimatorType = EstimatorType.Unknown;
        }

        internal Estimator(EstimatorType estimatorType)
            : this()
        {
            this.EstimatorType = estimatorType;
        }

        public EstimatorType EstimatorType { get; set; }

        public Parameter Parameter { get; set; }

        public static Entity operator +(Estimator left, Estimator right)
        {
            return new OneOfEntity()
            {
                Left = new EstimatorEntity(left),
                Right = new EstimatorEntity(right),
            };
        }

        public static Entity operator +(Entity left, Estimator right)
        {
            return new OneOfEntity()
            {
                Left = left,
                Right = new EstimatorEntity(right),
            };
        }

        public static Entity operator +(Estimator left, Entity right)
        {
            return new OneOfEntity()
            {
                Left = new EstimatorEntity(left),
                Right = right,
            };
        }

        public static Entity operator *(Estimator left, Estimator right)
        {
            return new ConcatenateEntity()
            {
                Left = new EstimatorEntity(left),
                Right = new EstimatorEntity(right),
            };
        }

        public static Entity operator *(Entity left, Estimator right)
        {
            return new ConcatenateEntity()
            {
                Left = left,
                Right = new EstimatorEntity(right),
            };
        }

        public static Entity operator *(Estimator left, Entity right)
        {
            return new ConcatenateEntity()
            {
                Left = new EstimatorEntity(left),
                Right = right,
            };
        }
    }
}
