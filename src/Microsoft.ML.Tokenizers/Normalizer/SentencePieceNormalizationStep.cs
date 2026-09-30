// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace Microsoft.ML.Tokenizers
{
    /// <summary>
    /// Applies a Hugging Face <c>tokenizer.json</c> normalizer chain in managed code so that JSON-only
    /// SentencePiece (Unigram) tokenizers normalize identically to the reference implementation before
    /// the SentencePiece model performs Metaspace pre-tokenization and Unigram decoding.
    /// </summary>
    /// <remarks>
    /// SentencePiece's own <see cref="SentencePieceNormalizer"/> fuses the precompiled charsmap with the
    /// Metaspace whitespace handling in a single pass and cannot represent arbitrary normalizer steps
    /// (per-character <c>Replace</c>, <c>Lowercase</c>, Unicode normalization, ...). When such steps are
    /// present this chain runs first (charsmap included), then the model's normalizer performs only the
    /// Metaspace escaping — matching the Hugging Face order (normalizer chain, then Metaspace pre-tokenizer).
    /// <para>
    /// Tradeoff: this is intentionally a simple string pipeline that allocates one intermediate string per
    /// step. The steps rely on string-only BCL APIs (<see cref="Regex"/>, <see cref="string.Normalize(NormalizationForm)"/>,
    /// <see cref="string.ToLowerInvariant"/>), so a span-based rewrite could only remove a subset of those
    /// allocations and was judged not to earn its complexity here: this chain runs only for the minority of
    /// models whose normalizer has content-modifying steps; the common path (a bare charsmap, and every
    /// protobuf-loaded model) sets <c>NormalizationChain</c> to null and bypasses it entirely. The caller
    /// decodes/encodes at the boundary through pooled buffers to avoid intermediate <c>byte[]</c> garbage.
    /// </para>
    /// </remarks>
    internal abstract class SentencePieceNormalizationStep
    {
        /// <summary>Applies this normalization step (or chain of steps) to <paramref name="text"/>.</summary>
        public abstract string Normalize(string text);

        // string.Normalize / NFD throw on lone (unpaired) surrogates, which a caller-supplied string may legitimately
        // contain. Replace any unpaired surrogate with U+FFFD so Unicode normalization degrades gracefully.
        private static string ReplaceLoneSurrogates(string text)
        {
            StringBuilder? sb = null;
            for (int i = 0; i < text.Length; i++)
            {
                char c = text[i];
                bool lone;
                if (char.IsHighSurrogate(c))
                {
                    lone = i + 1 >= text.Length || !char.IsLowSurrogate(text[i + 1]);
                }
                else if (char.IsLowSurrogate(c))
                {
                    lone = i == 0 || !char.IsHighSurrogate(text[i - 1]);
                }
                else
                {
                    lone = false;
                }

                if (lone)
                {
                    sb ??= new StringBuilder(text, 0, i, text.Length);
                    sb.Append('\uFFFD');
                }
                else
                {
                    sb?.Append(c);
                }
            }

            return sb is null ? text : sb.ToString();
        }

        /// <summary>
        /// Returns true if the normalizer tree contains a content-modifying step that the charsmap plus
        /// <c>removeExtraWhitespaces</c> approximation in <see cref="SentencePieceNormalizer"/> cannot represent
        /// (e.g. a literal/character <c>Replace</c>, <c>Lowercase</c>, Unicode normalization, or <c>Prepend</c>).
        /// A bare precompiled charsmap, a <c>Strip</c>, or a whitespace-collapsing <c>Replace</c> are not "rich".
        /// </summary>
        public static bool HasRichSteps(JsonElement normalizer)
        {
            if (normalizer.ValueKind != JsonValueKind.Object || !normalizer.TryGetProperty("type", out JsonElement typeElement) ||
                typeElement.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            string? type = typeElement.GetString();
            switch (type)
            {
                case "Sequence":
                    if (normalizer.TryGetProperty("normalizers", out JsonElement steps) && steps.ValueKind == JsonValueKind.Array)
                    {
                        foreach (JsonElement step in steps.EnumerateArray())
                        {
                            if (HasRichSteps(step))
                            {
                                return true;
                            }
                        }
                    }
                    return false;

                case "Precompiled":
                case "Strip":
                    return false;

                case "Replace":
                    // A whitespace-collapsing Replace is already approximated by removeExtraWhitespaces; any other
                    // Replace (literal punctuation splitting, character substitution) is content-modifying.
                    return !ReplaceIsWhitespaceCollapse(normalizer);

                default:
                    // Lowercase, NFC/NFD/NFKC/NFKD, Nmt, Prepend, ... all change content.
                    return true;
            }
        }

        /// <summary>
        /// Builds the managed normalizer chain from a <c>tokenizer.json</c> normalizer element. Throws
        /// <see cref="NotSupportedException"/> for normalizer types that are not modeled so callers fail loudly
        /// rather than silently mis-tokenizing.
        /// </summary>
        public static SentencePieceNormalizationStep Build(JsonElement normalizer)
        {
            if (normalizer.ValueKind != JsonValueKind.Object)
            {
                throw new InvalidDataException("A tokenizer.json normalizer entry must be a JSON object.");
            }

            string? type = normalizer.TryGetProperty("type", out JsonElement typeElement) && typeElement.ValueKind == JsonValueKind.String
                ? typeElement.GetString() : null;
            switch (type)
            {
                case "Sequence":
                    var children = new List<SentencePieceNormalizationStep>();
                    if (normalizer.TryGetProperty("normalizers", out JsonElement steps) && steps.ValueKind == JsonValueKind.Array)
                    {
                        foreach (JsonElement step in steps.EnumerateArray())
                        {
                            children.Add(Build(step));
                        }
                    }
                    return new SequenceStep(children);

                case "Precompiled":
                    {
                        string? charsMap = null;
                        if (normalizer.TryGetProperty("precompiled_charsmap", out JsonElement mapElement))
                        {
                            if (mapElement.ValueKind != JsonValueKind.String && mapElement.ValueKind != JsonValueKind.Null)
                            {
                                throw new InvalidDataException("The Precompiled normalizer 'precompiled_charsmap' must be a string.");
                            }

                            charsMap = mapElement.GetString();
                        }

                        return new PrecompiledStep(string.IsNullOrEmpty(charsMap) ? default : DecodePrecompiledCharsMap(charsMap!));
                    }

                case "Replace":
                    return ReplaceStep.Create(normalizer);

                case "Strip":
                    return new StripStep(
                        stripLeft: !normalizer.TryGetProperty("strip_left", out JsonElement left) || left.ValueKind != JsonValueKind.False,
                        stripRight: !normalizer.TryGetProperty("strip_right", out JsonElement right) || right.ValueKind != JsonValueKind.False);

                case "Lowercase":
                    return LowercaseStep.Instance;

                case "StripAccents":
                    return StripAccentsStep.Instance;

                case "NFC":
                    return new UnicodeStep(NormalizationForm.FormC);
                case "NFD":
                    return new UnicodeStep(NormalizationForm.FormD);
                case "NFKC":
                    return new UnicodeStep(NormalizationForm.FormKC);
                case "NFKD":
                    return new UnicodeStep(NormalizationForm.FormKD);

                case "Prepend":
                    {
                        if (normalizer.TryGetProperty("prepend", out JsonElement prependElement) &&
                            prependElement.ValueKind != JsonValueKind.String && prependElement.ValueKind != JsonValueKind.Null)
                        {
                            throw new InvalidDataException("The Prepend normalizer 'prepend' must be a string.");
                        }

                        string prepend = prependElement.ValueKind == JsonValueKind.String ? prependElement.GetString() ?? "" : "";
                        return new PrependStep(prepend);
                    }

                case "Nmt":
                    return NmtStep.Instance;

                default:
                    throw new NotSupportedException(
                        $"Unigram normalizer type '{type ?? "<missing>"}' is not supported when loading a tokenizer.json with content-modifying normalizer steps.");
            }
        }

        // Decodes a base64 'precompiled_charsmap' value, surfacing malformed input as InvalidDataException so callers
        // get a consistent, diagnostic failure for bad tokenizer.json files instead of a raw FormatException.
        internal static byte[] DecodePrecompiledCharsMap(string base64)
        {
            try
            {
                return Convert.FromBase64String(base64);
            }
            catch (FormatException ex)
            {
                throw new InvalidDataException("The tokenizer.json normalizer 'precompiled_charsmap' is not valid base64.", ex);
            }
        }

        // Mirrors SentencePieceTokenizer.ReplaceCollapsesSpaces: a Replace whose Regex matches runs of spaces.
        private static bool ReplaceIsWhitespaceCollapse(JsonElement replace)
        {
            if (!replace.TryGetProperty("pattern", out JsonElement patternElement) ||
                patternElement.ValueKind != JsonValueKind.Object ||
                !patternElement.TryGetProperty("Regex", out JsonElement regexElement) ||
                regexElement.ValueKind != JsonValueKind.String)
            {
                return false;
            }

            switch (regexElement.GetString())
            {
                case " {2,}":
                case " +":
                case "[ ]+":
                case "[ ]{2,}":
                case "\\s+":
                case "\\s{2,}":
                    return true;
                default:
                    return false;
            }
        }

        private sealed class SequenceStep : SentencePieceNormalizationStep
        {
            private readonly IReadOnlyList<SentencePieceNormalizationStep> _steps;

            public SequenceStep(IReadOnlyList<SentencePieceNormalizationStep> steps) => _steps = steps;

            public override string Normalize(string text)
            {
                foreach (SentencePieceNormalizationStep step in _steps)
                {
                    text = step.Normalize(text);
                }
                return text;
            }
        }

        private sealed class LowercaseStep : SentencePieceNormalizationStep
        {
            public static readonly LowercaseStep Instance = new LowercaseStep();
            public override string Normalize(string text) => text.ToLowerInvariant();
        }

        // Port of the Hugging Face tokenizers "StripAccents" normalizer: decompose (NFD) and drop combining marks.
        private sealed class StripAccentsStep : SentencePieceNormalizationStep
        {
            public static readonly StripAccentsStep Instance = new StripAccentsStep();

            public override string Normalize(string text)
            {
                if (text.Length == 0)
                {
                    return text;
                }

                string decomposed = ReplaceLoneSurrogates(text).Normalize(NormalizationForm.FormD);
                StringBuilder? sb = null;
                int i = 0;
                while (i < decomposed.Length)
                {
                    // Iterate by code point so combining marks encoded as surrogate pairs (astral plane) are
                    // classified and dropped, matching the reference (which filters on full code points).
                    int charCount = char.IsHighSurrogate(decomposed[i]) && i + 1 < decomposed.Length && char.IsLowSurrogate(decomposed[i + 1]) ? 2 : 1;
                    if (CharUnicodeInfo.GetUnicodeCategory(decomposed, i) == UnicodeCategory.NonSpacingMark)
                    {
                        if (sb is null)
                        {
                            sb = new StringBuilder(decomposed.Length);
                            sb.Append(decomposed, 0, i);
                        }
                    }
                    else
                    {
                        sb?.Append(decomposed, i, charCount);
                    }

                    i += charCount;
                }

                return sb is null ? decomposed : sb.ToString();
            }
        }

        private sealed class UnicodeStep : SentencePieceNormalizationStep
        {
            private readonly NormalizationForm _form;
            public UnicodeStep(NormalizationForm form) => _form = form;
            public override string Normalize(string text) => text.Length == 0 ? text : ReplaceLoneSurrogates(text).Normalize(_form);
        }

        private sealed class PrependStep : SentencePieceNormalizationStep
        {
            private readonly string _prefix;
            public PrependStep(string prefix) => _prefix = prefix;
            public override string Normalize(string text) => _prefix.Length == 0 ? text : _prefix + text;
        }

        // Port of the Hugging Face tokenizers "Nmt" normalizer: drop a set of control characters and map a set of
        // whitespace/format characters to a regular space. A no-op for text without those characters.
        private sealed class NmtStep : SentencePieceNormalizationStep
        {
            public static readonly NmtStep Instance = new NmtStep();

            private static bool IsRemoved(char c) =>
                (c >= '\u0001' && c <= '\u0008') || c == '\u000B' || (c >= '\u000E' && c <= '\u001F') ||
                c == '\u007F' || c == '\u008F' || c == '\u009F';

            private static bool IsSpace(char c) =>
                c == '\u0009' || c == '\u000A' || c == '\u000C' || c == '\u000D' || c == '\u1680' ||
                (c >= '\u200B' && c <= '\u200F') || c == '\u2028' || c == '\u2029' ||
                c == '\u2581' || c == '\uFEFF' || c == '\uFFFD';

            public override string Normalize(string text)
            {
                StringBuilder? sb = null;
                for (int i = 0; i < text.Length; i++)
                {
                    char c = text[i];
                    if (IsRemoved(c))
                    {
                        sb ??= NewBuilder(text, i);
                        continue;
                    }

                    char mapped = IsSpace(c) ? ' ' : c;
                    if (sb is not null)
                    {
                        sb.Append(mapped);
                    }
                    else if (mapped != c)
                    {
                        sb = NewBuilder(text, i);
                        sb.Append(mapped);
                    }
                }

                return sb is null ? text : sb.ToString();
            }

            private static StringBuilder NewBuilder(string text, int upTo)
            {
                var sb = new StringBuilder(text.Length);
                sb.Append(text, 0, upTo);
                return sb;
            }
        }

        private sealed class StripStep : SentencePieceNormalizationStep
        {
            private readonly bool _stripLeft;
            private readonly bool _stripRight;

            public StripStep(bool stripLeft, bool stripRight)
            {
                _stripLeft = stripLeft;
                _stripRight = stripRight;
            }

            public override string Normalize(string text)
            {
                if (_stripLeft && _stripRight)
                {
                    return text.Trim();
                }
                if (_stripLeft)
                {
                    return text.TrimStart();
                }
                return _stripRight ? text.TrimEnd() : text;
            }
        }

        private sealed class ReplaceStep : SentencePieceNormalizationStep
        {
            // Bounds worst-case evaluation of regexes parsed from untrusted tokenizer.json.
            private static readonly TimeSpan _regexTimeout = TimeSpan.FromSeconds(1);

            private readonly string _content;
            private readonly string? _literal;
            private readonly Regex? _regex;

            private ReplaceStep(string? literal, Regex? regex, string content)
            {
                _literal = literal;
                _regex = regex;
                _content = content;
            }

            public static ReplaceStep Create(JsonElement normalizer)
            {
                string content = normalizer.TryGetProperty("content", out JsonElement contentElement) && contentElement.ValueKind == JsonValueKind.String
                    ? contentElement.GetString() ?? "" : "";
                if (!normalizer.TryGetProperty("pattern", out JsonElement pattern) || pattern.ValueKind != JsonValueKind.Object)
                {
                    throw new InvalidDataException("Replace normalizer is missing its pattern.");
                }

                if (pattern.TryGetProperty("String", out JsonElement literal))
                {
                    if (literal.ValueKind != JsonValueKind.String)
                    {
                        throw new InvalidDataException("Replace normalizer 'String' pattern must be a string.");
                    }

                    return new ReplaceStep(literal.GetString() ?? "", regex: null, content);
                }

                if (pattern.TryGetProperty("Regex", out JsonElement regex))
                {
                    if (regex.ValueKind != JsonValueKind.String)
                    {
                        throw new InvalidDataException("Replace normalizer 'Regex' pattern must be a string.");
                    }

                    string regexPattern = regex.GetString()!;
                    try
                    {
                        return new ReplaceStep(literal: null, new Regex(regexPattern, RegexOptions.CultureInvariant, _regexTimeout), content);
                    }
                    catch (ArgumentException ex)
                    {
                        throw new InvalidDataException($"Replace normalizer has an invalid Regex pattern '{regexPattern}'.", ex);
                    }
                }

                throw new NotSupportedException("Replace normalizer requires a String or Regex pattern.");
            }

            public override string Normalize(string text)
            {
                if (_regex is not null)
                {
                    // Hugging Face replaces the matched range with 'content' literally; escape '$' so Regex.Replace
                    // does not interpret it as a substitution pattern (e.g. "$0", "$&") and diverge from the reference.
                    string replacement = _content.IndexOf('$') < 0 ? _content : _content.Replace("$", "$$");
                    return _regex.Replace(text, replacement);
                }

                return string.IsNullOrEmpty(_literal) ? text : text.Replace(_literal, _content);
            }
        }

        // Applies a SentencePiece precompiled charsmap by delegating to a charsmap-only SentencePieceNormalizer
        // (no dummy prefix, no whitespace escaping, no whitespace stripping), reusing the existing DARTS engine.
        private sealed class PrecompiledStep : SentencePieceNormalizationStep
        {
            private readonly SentencePieceNormalizer? _charsMap;

            public PrecompiledStep(ReadOnlySpan<byte> precompiledCharsMap)
            {
                if (!precompiledCharsMap.IsEmpty)
                {
                    _charsMap = new SentencePieceNormalizer(
                        precompiledCharsMap,
                        removeExtraWhiteSpaces: false,
                        addDummyPrefix: false,
                        escapeWhiteSpaces: false,
                        treatWhitespaceAsSuffix: false,
                        specialTokens: null);
                }
            }

            public override string Normalize(string text) => _charsMap is null ? text : _charsMap.NormalizeUtf8ToString(text);
        }
    }
}
