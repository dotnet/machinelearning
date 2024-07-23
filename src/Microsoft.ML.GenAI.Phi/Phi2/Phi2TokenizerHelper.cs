// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML.Tokenizers;
using Tensorboard;

/// <summary>
/// The utility class to create tokenizer for phi-3 model.
/// </summary>
public class Phi2TokenizerHelper
{
    public static CodeGenTokenizer Create(
        string folder,
        string vocabFile = "vocab.json",
        string mergesFile = "merges.txt",
        bool addPrefixSpace = false,
        bool addBeginOfSentence = false,
        bool addEndOfSentence = false)
    {
        var vocabPath = Path.Combine(folder, vocabFile);
        var mergesPath = Path.Combine(folder, mergesFile);
        using var vocabStream = File.OpenRead(vocabPath);
        using var mergesStream = File.OpenRead(mergesPath);

        return CodeGenTokenizer.Create(vocabStream, mergesStream, addPrefixSpace, addBeginOfSentence, addEndOfSentence);
    }
}
