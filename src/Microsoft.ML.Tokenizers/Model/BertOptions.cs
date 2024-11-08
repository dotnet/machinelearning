// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Options for the Bert tokenizer.
    /// </summary>
    public sealed class BertOptions : WordPieceOptions
    {
#pragma warning disable MSML_NoInstanceInitializers
        /// <summary>
        /// Gets or sets a value indicating whether to lower case the input before tokenization.
        /// </summary>
        public bool LowerCaseBeforeTokenization { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to apply basic tokenization.
        /// </summary>
        public bool ApplyBasicTokenization { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to split on special tokens.
        /// </summary>
        public bool SplitOnSpecialTokens { get; set; } = true;

        /// <summary>
        /// Gets or sets the separator token to use.
        /// </summary>
        public string SeparatorToken { get; set; } = "[SEP]";

        /// <summary>
        /// Gets or sets the padding token to use.
        /// </summary>
        public string PaddingToken { get; set; } = "[PAD]";

        /// <summary>
        /// Gets or sets the classification token to use.
        /// </summary>
        public string ClassificationToken { get; set; } = "[CLS]";

        /// <summary>
        /// Gets or sets the masking token to use.
        /// </summary>
        public string MaskingToken { get; set; } = "[MASK]";

        /// <summary>
        /// Gets or sets a value indicating whether to tokenize the CJK characters in separate tokens.
        /// </summary>
        /// <remarks>
        /// This is useful when you want to tokenize CJK characters individually.
        /// The following Unicode ranges are considered CJK characters for this purpose:
        /// - U+3400 - U+4DBF   CJK Unified Ideographs Extension A.
        /// - U+4E00 - U+9FFF   basic set of CJK characters.
        /// - U+F900 - U+FAFF   CJK Compatibility Ideographs.
        /// - U+20000 - U+2A6DF CJK Unified Ideographs Extension B.
        /// - U+2A700 - U+2B73F CJK Unified Ideographs Extension C.
        /// - U+2B740 - U+2B81F CJK Unified Ideographs Extension D.
        /// - U+2B820 - U+2CEAF CJK Unified Ideographs Extension E.
        /// - U+2F800 - U+2FA1F CJK Compatibility Ideographs Supplement.
        /// </remarks>
        public bool IndividuallyTokenizeCjk { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether to remove non-spacing marks.
        /// </summary>
        public bool RemoveNonSpacingMarks { get; set; }

#pragma warning restore MSML_NoInstanceInitializers
    }
}