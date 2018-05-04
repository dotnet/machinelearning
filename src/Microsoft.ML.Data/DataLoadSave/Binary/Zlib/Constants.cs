// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.Runtime.Data.IO.Zlib
{
    /// <summary>
    /// See zlib.h
    /// </summary>
    public static class Constants
    {
        /// <summary>
        /// Maximum size of history buffer inside zlib.
        /// </summary>
        public const int MaxBufferSize = 15;

        public enum Flush
        {
            NoFlush = 0,
            PartialFlush = 1,
            SyncFlush = 2,
            FullFlush = 3,
            Finish = 4,
            Block = 5,
            Trees = 6,
        };

        public enum RetCode
        {
            VersionError = -6,
            BufError = -5,
            MemError = -4,
            DataError = -3,
            StreamError = -2,
            Errno = -1,
            OK = 0,
            StreamEnd = 1,
            NeedDict = 2,
        }

        public enum Level
        {
            DefaultCompression = -1,
            Level0 = 0,
            NoCompression = 0,
            BestSpeed = 1,
            Level1 = 1,
            Level2 = 2,
            Level3 = 3,
            Level4 = 4,
            Level5 = 5,
            Level6 = 6,
            Level7 = 7,
            Level8 = 8,
            BestCompression = 9,
            Level9 = 9,
        }

        public enum Strategy
        {
            DefaultStrategy = 0,
            Filtered = 1,
            HuffmanOnly = 2,
            Rle = 3,
            Fixed = 4,
        }

        public enum Type
        {
            Binary = 0,
            Ascii = 1,
            Text = 1,
            Unknown = 2,
        }
    }
}
