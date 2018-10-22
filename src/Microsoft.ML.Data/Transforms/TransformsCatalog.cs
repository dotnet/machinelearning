// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Microsoft.ML.Runtime
{
    /// <summary>
    /// Similar to training context, a transform context is an object serving as a 'catalog' of available transforms.
    /// Individual transforms are exposed as extension methods of this class or its subclasses.
    /// </summary>
    public sealed class TransformsCatalog
    {
        internal IHostEnvironment Environment { get; }

        public CategoricalTransforms Categorical { get; }
        public Conversions Conversion { get; }
        public TextTransforms Text { get; }
        public ProjectionTransforms Projections { get; }

        internal TransformsCatalog(IHostEnvironment env)
        {
            Contracts.AssertValue(env);
            Environment = env;

            Categorical = new CategoricalTransforms(this);
            Conversion = new Conversions(this);
            Text = new TextTransforms(this);
            Projections = new ProjectionTransforms(this);
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
        /// The catalog of rescaling operations.
        /// </summary>
        public sealed class Conversions : SubCatalogBase
        {
            public Conversions(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of text processing operations.
        /// </summary>
        public sealed class TextTransforms : SubCatalogBase
        {
            public TextTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }

        /// <summary>
        /// The catalog of projection operations.
        /// </summary>
        public sealed class ProjectionTransforms : SubCatalogBase
        {
            public ProjectionTransforms(TransformsCatalog owner) : base(owner)
            {
            }
        }
    }
}
