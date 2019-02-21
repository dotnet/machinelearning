// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML
{
    /// <summary>
    /// Similar to training context, a transform context is an object serving as a 'catalog' of available transforms.
    /// Individual transforms are exposed as extension methods of this class or its subclasses.
    /// </summary>
    public sealed class TransformsCatalog
    {
        internal IHostEnvironment Environment { get; }

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
        /// The list of operations for data projection.
        /// </summary>
        public ProjectionTransforms Projection { get; }

        /// <summary>
        /// The list of operations for selecting features based on some criteria.
        /// </summary>
        public FeatureSelectionTransforms FeatureSelection { get; }

        internal TransformsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;

            Categorical = new CategoricalTransforms(this);
            Conversion = new ConversionTransforms(this);
            Text = new TextTransforms(this);
            Projection = new ProjectionTransforms(this);
            FeatureSelection = new FeatureSelectionTransforms(this);
        }

        public abstract class SubCatalogBase
        {
            internal IHostEnvironment Environment { get; }

            protected SubCatalogBase(TransformsCatalog owner)
            {
                Environment = owner.Environment;
            }

        }

        /// <summary>
        /// The catalog of operations over categorical data.
        /// </summary>
        public sealed class CategoricalTransforms : SubCatalogBase
        {
            internal CategoricalTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of type conversion operations.
        /// </summary>
        public sealed class ConversionTransforms : SubCatalogBase
        {
            internal ConversionTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of text processing operations.
        /// </summary>
        public sealed class TextTransforms : SubCatalogBase
        {
            internal TextTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of projection operations.
        /// </summary>
        public sealed class ProjectionTransforms : SubCatalogBase
        {
            internal ProjectionTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of feature selection operations.
        /// </summary>
        public sealed class FeatureSelectionTransforms : SubCatalogBase
        {
            internal FeatureSelectionTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }
    }
}
