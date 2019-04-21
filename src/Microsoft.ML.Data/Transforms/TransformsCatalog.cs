// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;

namespace Microsoft.ML
{
    /// <summary>
    /// Class used by <see cref="MLContext"/> to create instances of transform components.
    /// </summary>
    public sealed class TransformsCatalog : IInternalCatalog
    {
        IHostEnvironment IInternalCatalog.Environment => _env;
        private readonly IHostEnvironment _env;

        /// <summary>
        /// The list of operations over categorical data.
        /// </summary>
        public CategoricalTransforms Categorical { get; }

        /// <summary>
        /// The list of operations for data type conversion.
        /// </summary>
        public ConversionTransforms Conversion { get; }

        /// <summary>
        /// The list of operations for processing text data.
        /// </summary>
        public TextTransforms Text { get; }

        /// <summary>
        /// The list of operations for selecting features based on some criteria.
        /// </summary>
        public FeatureSelectionTransforms FeatureSelection { get; }

        internal TransformsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            _env = env;

            Categorical = new CategoricalTransforms(this);
            Conversion = new ConversionTransforms(this);
            Text = new TextTransforms(this);
            FeatureSelection = new FeatureSelectionTransforms(this);
        }

        /// <summary>
        /// Class used by <see cref="MLContext"/> to create instances of categorical data transform components.
        /// </summary>
        public sealed class CategoricalTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal CategoricalTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }

        /// <summary>
        /// Class used by <see cref="MLContext"/> to create instances of type conversion data transform components.
        /// </summary>
        public sealed class ConversionTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal ConversionTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }

        /// <summary>
        /// Class used by <see cref="MLContext"/> to create instances of text data transform components.
        /// </summary>
        public sealed class TextTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal TextTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }

        /// <summary>
        /// Class used by <see cref="MLContext"/> to create instances of feature selection transform components.
        /// </summary>
        public sealed class FeatureSelectionTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal FeatureSelectionTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }
    }
}
