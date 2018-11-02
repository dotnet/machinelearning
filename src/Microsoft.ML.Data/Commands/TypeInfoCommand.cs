// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data.Commands;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Command;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Data.Conversion;
using Microsoft.ML.Runtime.Internal.Utilities;

[assembly: LoadableClass(typeof(TypeInfoCommand), typeof(TypeInfoCommand.Arguments), typeof(SignatureCommand),
    "", TypeInfoCommand.LoadName)]

namespace Microsoft.ML.Data.Commands
{
    public sealed class TypeInfoCommand : ICommand
    {
        internal const string LoadName = "TypeInfo";
        internal const string Summary = "Displays information about the standard primitive " +
            "non-key types, and conversions between them.";

        public sealed class Arguments
        {
        }

        private readonly IHost _host;

        public TypeInfoCommand(IHostEnvironment env, Arguments args)
        {
            Contracts.CheckValue(env, nameof(env));
            _host = env.Register(LoadName);
            _host.CheckValue(args, nameof(args));
        }

        private readonly struct TypeNaInfo
        {
            public readonly bool HasNa;
            public readonly bool DefaultIsNa;

            public TypeNaInfo(bool hasNa, bool defaultIsNa)
            {
                HasNa = hasNa;
                DefaultIsNa = defaultIsNa;
            }
        }

        private sealed class SetOfKindsComparer : IEqualityComparer<ISet<DataKind>>
        {
            public bool Equals(ISet<DataKind> x, ISet<DataKind> y)
            {
                Contracts.AssertValueOrNull(x);
                Contracts.AssertValueOrNull(y);
                if (x == null || y == null)
                    return (x == null) && (y == null);
                return x.SetEquals(y);
            }

            public int GetHashCode(ISet<DataKind> obj)
            {
                Contracts.AssertValueOrNull(obj);
                int hash = 0;
                if (obj != null)
                {
                    foreach (var kind in obj.OrderBy(x => x))
                        hash = Hashing.CombineHash(hash, kind.GetHashCode());
                }
                return hash;
            }
        }

        public void Run()
        {
            using (var ch = _host.Start("Run"))
            {
                var conv = Conversions.Instance;
                var comp = new SetOfKindsComparer();
                var dstToSrcMap = new Dictionary<HashSet<DataKind>, HashSet<DataKind>>(comp);
                var srcToDstMap = new Dictionary<DataKind, HashSet<DataKind>>();

                var kinds = Enum.GetValues(typeof(DataKind)).Cast<DataKind>().Distinct().OrderBy(k => k).ToArray();
                var types = kinds.Select(kind => PrimitiveType.FromKind(kind)).ToArray();

                HashSet<DataKind> nonIdentity = null;
                // For each kind and its associated type.
                for (int i = 0; i < types.Length; ++i)
                {
                    ch.AssertValue(types[i]);
                    var info = Utils.MarshalInvoke(KindReport<int>, types[i].RawType, ch, types[i]);

                    var dstKinds = new HashSet<DataKind>();
                    Delegate del;
                    bool isIdentity;
                    for (int j = 0; j < types.Length; ++j)
                    {
                        if (conv.TryGetStandardConversion(types[i], types[j], out del, out isIdentity))
                            dstKinds.Add(types[j].RawKind);
                    }
                    if (!conv.TryGetStandardConversion(types[i], types[i], out del, out isIdentity))
                        Utils.Add(ref nonIdentity, types[i].RawKind);
                    else
                        ch.Assert(isIdentity);

                    srcToDstMap[types[i].RawKind] = dstKinds;
                    HashSet<DataKind> srcKinds;
                    if (!dstToSrcMap.TryGetValue(dstKinds, out srcKinds))
                        dstToSrcMap[dstKinds] = srcKinds = new HashSet<DataKind>();
                    srcKinds.Add(types[i].RawKind);
                }

                // Now perform the final outputs.
                for (int i = 0; i < kinds.Length; ++i)
                {
                    var dsts = srcToDstMap[kinds[i]];
                    HashSet<DataKind> srcs;
                    if (!dstToSrcMap.TryGetValue(dsts, out srcs))
                        continue;
                    ch.Assert(Utils.Size(dsts) >= 1);
                    ch.Assert(Utils.Size(srcs) >= 1);
                    string srcStrings = string.Join(", ", srcs.OrderBy(k => k).Select(k => '`' + k.GetString() + '`'));
                    string dstStrings = string.Join(", ", dsts.OrderBy(k => k).Select(k => '`' + k.GetString() + '`'));
                    dstToSrcMap.Remove(dsts);
                    ch.Info(srcStrings + " | " + dstStrings);
                }

                if (Utils.Size(nonIdentity) > 0)
                {
                    ch.Warning("The following kinds did not have an identity conversion: {0}",
                        string.Join(", ", nonIdentity.OrderBy(k => k).Select(DataKindExtensions.GetString)));
                }
            }
        }

        private TypeNaInfo KindReport<T>(IChannel ch, PrimitiveType type)
        {
            Contracts.AssertValue(ch);
            ch.AssertValue(type);
            ch.Assert(type.IsStandardScalar);

            var conv = Conversions.Instance;
            InPredicate<T> isNaDel;
            bool hasNaPred = conv.TryGetIsNAPredicate(type, out isNaDel);
            bool defaultIsNa = false;
            if (hasNaPred)
            {
                T def = default(T);
                defaultIsNa = isNaDel(in def);
            }
            return new TypeNaInfo(hasNaPred, defaultIsNa);
        }
    }
}
