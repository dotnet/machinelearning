// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;

namespace Microsoft.ML
{
    /// <summary>
    /// Similar to training context, a transform context is an object serving as a 'catalog' of available transforms.
    /// Individual transforms are exposed as extension methods of this class or its subclasses.
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
        /// The list of operations for applying kernel methods.
        /// </summary>
        public KernelTransforms Kernel { get; }

        /// <summary>
        /// The list of operations for data normalization.
        /// </summary>
        public NormalizationTransforms Normalization { get; }

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
            Kernel = new KernelTransforms(this);
            FeatureSelection = new FeatureSelectionTransforms(this);
            Normalization = new NormalizationTransforms(this);
        }

        /// <summary>
        /// The catalog of operations over categorical data.
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
        /// The catalog of type conversion operations.
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
        /// The catalog of text processing operations.
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
        /// The catalog of kernel methods.
        /// </summary>
        public sealed class KernelTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal KernelTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }

        /// <summary>
        /// The catalog of feature selection operations.
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

        /// <summary>
        /// The catalog of normalization.
        /// </summary>
        public sealed class NormalizationTransforms : IInternalCatalog
        {
            IHostEnvironment IInternalCatalog.Environment => _env;
            private readonly IHostEnvironment _env;

            internal NormalizationTransforms(TransformsCatalog owner)
            {
                _env = owner.GetEnvironment();
            }
        }
    }
}
