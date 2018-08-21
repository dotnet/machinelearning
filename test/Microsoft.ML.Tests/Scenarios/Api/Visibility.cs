// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

﻿using Microsoft.ML.Runtime.Data;
using System;
using Xunit;

namespace Microsoft.ML.Tests.Scenarios.Api
{
    public partial class ApiScenariosTests
    {
        /// <summary>
        /// Visibility: It should, possibly through the debugger, be not such a pain to actually
        /// see what is happening to your data when you apply this or that transform. E.g.: if I
        /// were to have the text "Help I'm a bug!" I should be able to see the steps where it is
        /// normalized to "help i'm a bug" then tokenized into ["help", "i'm", "a", "bug"] then
        /// mapped into term numbers [203, 25, 3, 511] then projected into the sparse
        /// float vector {3:1, 25:1, 203:1, 511:1}, etc. etc.
        /// </summary>
        [Fact]
        void Visibility()
        {
            var dataPath = GetDataPath(SentimentDataPath);
            var testDataPath = GetDataPath(SentimentTestPath);

            using (var env = new TlcEnvironment(seed: 1, conc: 1))
            {
                // Pipeline.
                var loader = new TextLoader(env, MakeSentimentTextLoaderArgs(), new MultiFileSource(dataPath));

                var trans = TextTransform.Create(env, MakeSentimentTextTransformArgs(false), loader);
                
                // In order to find out available column names, you can go through schema and check
                // column names and appropriate type for getter.
                for( int i=0; i< trans.Schema.ColumnCount; i++)
                {
                    var columnName = trans.Schema.GetColumnName(i);
                    var columnType = trans.Schema.GetColumnType(i).RawType;
                }

                using (var cursor = trans.GetRowCursor(x => true))
                {
                    Assert.True(cursor.Schema.TryGetColumnIndex("SentimentText", out int textColumn));
                    Assert.True(cursor.Schema.TryGetColumnIndex("Features_TransformedText", out int transformedTextColumn));
                    Assert.True(cursor.Schema.TryGetColumnIndex("Features", out int featureColumn));
                    
                    var originalTextGettter = cursor.GetGetter<ReadOnlyMemory<char>>(textColumn);
                    var transformedTextGettter = cursor.GetGetter<VBuffer<ReadOnlyMemory<char>>>(transformedTextColumn);
                    var featureGettter = cursor.GetGetter<VBuffer<float>>(featureColumn);
                    ReadOnlyMemory<char> text = default;
                    VBuffer<ReadOnlyMemory<char>> transformedText = default;
                    VBuffer<float> features = default;
                    while (cursor.MoveNext())
                    {
                        originalTextGettter(ref text);
                        transformedTextGettter(ref transformedText);
                        featureGettter(ref features);
                    }
                }
            }
        }
    }
}
