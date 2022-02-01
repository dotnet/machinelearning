// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using Microsoft.ML.Internal.Utilities;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using BL = System.Boolean;
    using I4 = System.Int32;
    using I8 = System.Int64;
    using R4 = Single;
    using R8 = Double;
    using TX = ReadOnlyMemory<char>;

    internal sealed partial class LambdaCompiler : IDisposable
    {
        public const int MaxParams = 16;

        private readonly LambdaNode _top;
        private readonly Type _delType;
        private readonly MethodGenerator _meth;

        public static Delegate Compile(out List<Error> errors, LambdaNode node)
        {
            Contracts.CheckValue(node, nameof(node));

            using (var cmp = new LambdaCompiler(node))
                return cmp.Do(out errors);
        }

        private LambdaCompiler(LambdaNode node)
        {
            Contracts.AssertValue(node);
            Contracts.Assert(1 <= node.Vars.Length && node.Vars.Length <= MaxParams);
            Contracts.Assert(MaxParams <= 16);
            Contracts.AssertValue(node.ResultType);

            _top = node;

            Type typeFn;
            switch (node.Vars.Length)
            {
                case 1:
                    { typeFn = typeof(Func<,>); break; }
                case 2:
                    { typeFn = typeof(Func<,,>); break; }
                case 3:
                    { typeFn = typeof(Func<,,,>); break; }
                case 4:
                    { typeFn = typeof(Func<,,,,>); break; }
                case 5:
                    { typeFn = typeof(Func<,,,,,>); break; }
                default:
                    throw Contracts.Except("Internal error in LambdaCompiler: Maximum number of inputs exceeded.");
            }

            var types = new Type[node.Vars.Length + 1];
            foreach (var v in node.Vars)
            {
                Contracts.Assert(0 <= v.Index && v.Index < node.Vars.Length);
                Contracts.Assert(types[v.Index] == null);
                types[v.Index] = v.Type.RawType;
            }
            types[node.Vars.Length] = node.ResultType.RawType;
            _delType = typeFn.MakeGenericType(types);

            Array.Copy(types, 0, types, 1, types.Length - 1);
            types[0] = typeof(object);

            _meth = new MethodGenerator("lambda", typeof(Exec), node.ResultType.RawType, types);
        }

        private Delegate Do(out List<Error> errors)
        {
            var visitor = new Visitor(_meth);
            _top.Expr.Accept(visitor);

            errors = visitor.GetErrors();
            if (errors != null)
                return null;

            _meth.Il.Ret();
            return _meth.CreateDelegate(_delType);
        }

        public void Dispose()
        {
            _meth.Dispose();
        }
    }

    internal sealed partial class LambdaCompiler
    {
        private sealed class Visitor : ExprVisitor
        {
            private static readonly MethodInfo _methGetFalseBL = ((Func<BL>)BuiltinFunctions.False).GetMethodInfo();
            private static readonly MethodInfo _methGetTrueBL = ((Func<BL>)BuiltinFunctions.True).GetMethodInfo();

            private readonly MethodGenerator _meth;
            private readonly ILGenerator _gen;
            private List<Error> _errors;

            private sealed class CachedWithLocal
            {
                public readonly WithLocalNode Node;
                /// <summary>
                /// The IL local containing the computed value.
                /// </summary>
                public readonly LocalBuilder Value;
                /// <summary>
                /// The boolean local indicating whether the value has been computed yet.
                /// If the value is pre-computed, this is null.
                /// </summary>
                public readonly LocalBuilder Flag;

                public CachedWithLocal(WithLocalNode node, LocalBuilder value, LocalBuilder flag)
                {
                    Contracts.AssertValue(node);
                    Contracts.AssertValue(value);
                    Contracts.AssertValueOrNull(flag);
                    Node = node;
                    Value = value;
                    Flag = flag;
                }
            }

            // The active cached "with" locals. For "with" local values that are aggressively computed, the
            // corresponding flag local is null. For lazy computed values, the flag indicates whether
            // the value has been computed and stored yet. Lazy computed values avoid potentially
            // expensive computation that might not be needed, but result in code bloat since each
            // use tests the flag, and if false, computes and stores the value.
            private readonly List<CachedWithLocal> _cacheWith;

            public Visitor(MethodGenerator meth)
            {
                _meth = meth;
                _gen = meth.Il;

                _cacheWith = new List<CachedWithLocal>();
            }

            public List<Error> GetErrors()
            {
                return _errors;
            }

            private void DoConvert(ExprNode node)
            {
                Contracts.AssertValue(node);
                if (!node.NeedsConversion)
                {
                    // When dealing with floating point, always emit the conversion so
                    // the result is stable.
                    if (node.IsR4)
                        _gen.Conv_R4();
                    else if (node.IsR8)
                        _gen.Conv_R8();
                    return;
                }

                switch (node.SrcKind)
                {
                    default:
                        Contracts.Assert(false, "Unexpected src kind in DoConvert");
                        PostError(node, "Internal error in implicit conversion");
                        break;

                    case ExprTypeKind.R4:
                        // R4 will only implicitly convert to R8.
                        if (node.IsR8)
                        {
                            _gen.Conv_R8();
                            return;
                        }
                        break;
                    case ExprTypeKind.I4:
                        // I4 can convert to I8, R4, or R8.
                        switch (node.ExprType)
                        {
                            case ExprTypeKind.I8:
                                _gen.Conv_I8();
                                return;
                            case ExprTypeKind.R4:
                                _gen.Conv_R4();
                                return;
                            case ExprTypeKind.R8:
                                _gen.Conv_R8();
                                return;
                        }
                        break;
                    case ExprTypeKind.I8:
                        // I8 will only implicitly convert to R8.
                        if (node.IsR8)
                        {
                            _gen.Conv_R8();
                            return;
                        }
                        break;
                }

                Contracts.Assert(false, "Unexpected dst kind in DoConvert");
                PostError(node, "Internal error(2) in implicit conversion");
            }

            private void PostError(Node node)
            {
                Utils.Add(ref _errors, new Error(node.Token, "Code generation error"));
            }

            private void PostError(Node node, string msg)
            {
                Utils.Add(ref _errors, new Error(node.Token, msg));
            }

            private void PostError(Node node, string msg, params object[] args)
            {
                Utils.Add(ref _errors, new Error(node.Token, msg, args));
            }

            private bool TryUseValue(ExprNode node)
            {
                var value = node.ExprValue;
                if (value == null)
                    return false;

                switch (node.ExprType)
                {
                    case ExprTypeKind.BL:
                        Contracts.Assert(value is BL);
                        GenBL((BL)value);
                        break;
                    case ExprTypeKind.I4:
                        Contracts.Assert(value is I4);
                        _gen.Ldc_I4((I4)value);
                        break;
                    case ExprTypeKind.I8:
                        Contracts.Assert(value is I8);
                        _gen.Ldc_I8((I8)value);
                        break;
                    case ExprTypeKind.R4:
                        Contracts.Assert(value is R4);
                        _gen.Ldc_R4((R4)value);
                        break;
                    case ExprTypeKind.R8:
                        Contracts.Assert(value is R8);
                        _gen.Ldc_R8((R8)value);
                        break;
                    case ExprTypeKind.TX:
                        {
                            Contracts.Assert(value is TX);
                            TX text = (TX)value;
                            _gen.Ldstr(text.ToString());
                            CallFnc<string, TX>(Exec.ToTX);
                        }
                        break;

                    case ExprTypeKind.Error:
                        PostError(node);
                        break;

                    default:
                        PostError(node, "Bad ExprType");
                        break;
                }

                return true;
            }

            public override void Visit(BoolLitNode node)
            {
                Contracts.Assert(node.IsBool);
                Contracts.Assert(node.ExprValue is BL);
                GenBL((BL)node.ExprValue);
            }

            public override void Visit(StrLitNode node)
            {
                Contracts.Assert(node.IsTX);
                Contracts.Assert(node.ExprValue is TX);
                TX text = (TX)node.ExprValue;

                _gen.Ldstr(text.ToString());
                CallFnc<string, TX>(Exec.ToTX);
            }

            public override void Visit(NumLitNode node)
            {
                Contracts.Assert(node.IsNumber);
                var value = node.ExprValue;
                Contracts.Assert(value != null);
                switch (node.ExprType)
                {
                    case ExprTypeKind.I4:
                        Contracts.Assert(value is I4);
                        _gen.Ldc_I4((I4)value);
                        break;
                    case ExprTypeKind.I8:
                        Contracts.Assert(value is I8);
                        _gen.Ldc_I8((I8)value);
                        break;
                    case ExprTypeKind.R4:
                        Contracts.Assert(value is R4);
                        _gen.Ldc_R4((R4)value);
                        break;
                    case ExprTypeKind.R8:
                        Contracts.Assert(value is R8);
                        _gen.Ldc_R8((R8)value);
                        break;
                    default:
                        Contracts.Assert(false, "Bad NumLitNode");
                        PostError(node, "Internal error in numeric literal");
                        break;
                }
            }

            public override void Visit(IdentNode node)
            {
                if (TryUseValue(node))
                    return;

                Node referent = node.Referent;
                if (node.Referent == null)
                {
                    PostError(node, "Unbound name!");
                    return;
                }

                switch (referent.Kind)
                {
                    default:
                        PostError(node, "Unbound name!");
                        return;

                    case NodeKind.Param:
                        _gen.Ldarg(referent.AsParam.Index + 1);
                        break;

                    case NodeKind.WithLocal:
                        var loc = referent.AsWithLocal;
                        Contracts.Assert(loc.Value.ExprValue == null);
                        Contracts.Assert(loc.GenCount >= 0);

                        if (loc.UseCount <= 1)
                        {
                            Contracts.Assert(loc.UseCount == 1);
                            Contracts.Assert(loc.Index == -1);
                            loc.GenCount++;
                            loc.Value.Accept(this);
                        }
                        else
                        {
                            Contracts.Assert(0 <= loc.Index && loc.Index < _cacheWith.Count);
                            var cache = _cacheWith[loc.Index];
                            Contracts.Assert(cache.Value != null);
                            if (cache.Flag != null)
                            {
                                // This is a lazy computed value. If we've already computed the value, skip the code
                                // that generates it. If this is the first place that generates it, we don't need to
                                // test the bool - we know it hasn't been computed yet (since we never jump backwards).
                                bool needTest = loc.GenCount > 0;
                                Label labHave = default(Label);
                                if (needTest)
                                {
                                    labHave = _gen.DefineLabel();
                                    _gen
                                        .Ldloc(cache.Flag)
                                        .Brtrue(labHave);
                                }

                                // Generate the code for the value.
                                loc.GenCount++;
                                loc.Value.Accept(this);

                                // Store the value and set the flag indicating that we have it.
                                _gen
                                    .Stloc(cache.Value)
                                    .Ldc_I4(1)
                                    .Stloc(cache.Flag);
                                if (needTest)
                                    _gen.MarkLabel(labHave);
                            }

                            // Load the value.
                            _gen.Ldloc(cache.Value);
                        }
                        break;
                }
                DoConvert(node);
            }

            public override bool PreVisit(UnaryOpNode node)
            {
                if (TryUseValue(node))
                    return false;
                return true;
            }

            public override void PostVisit(UnaryOpNode node)
            {
                Contracts.AssertValue(node);

                switch (node.Op)
                {
                    default:
                        Contracts.Assert(false, "Bad unary op");
                        PostError(node, "Internal error in unary operator");
                        break;

                    case UnaryOp.Minus:
                        Contracts.Assert(node.IsNumber);
                        Contracts.Assert(node.Arg.ExprType == node.SrcKind);
                        switch (node.SrcKind)
                        {
                            case ExprTypeKind.I4:
                                _gen.Neg();
                                break;
                            case ExprTypeKind.I8:
                                _gen.Neg();
                                break;
                            case ExprTypeKind.R4:
                            case ExprTypeKind.R8:
                                _gen.Neg();
                                break;

                            default:
                                Contracts.Assert(false, "Bad operand type in unary minus");
                                PostError(node, "Internal error in unary minus");
                                break;
                        }
                        break;

                    case UnaryOp.Not:
                        CallFnc<BL, byte>(BuiltinFunctions.Not);
                        break;
                }

                DoConvert(node);
            }

            public override bool PreVisit(BinaryOpNode node)
            {
                Contracts.AssertValue(node);

                if (TryUseValue(node))
                    return false;

                if (node.ReduceToLeft)
                {
                    node.Left.Accept(this);
                    DoConvert(node);
                    return false;
                }

                if (node.ReduceToRight)
                {
                    node.Right.Accept(this);
                    DoConvert(node);
                    return false;
                }

                switch (node.Op)
                {
                    default:
                        Contracts.Assert(false, "Bad binary op");
                        PostError(node, "Internal error in binary operator");
                        break;

                    case BinaryOp.Coalesce:
                        GenCoalesce(node);
                        break;

                    case BinaryOp.Or:
                    case BinaryOp.And:
                        GenBoolBinOp(node);
                        break;

                    case BinaryOp.Add:
                    case BinaryOp.Sub:
                    case BinaryOp.Mul:
                    case BinaryOp.Div:
                    case BinaryOp.Mod:
                    case BinaryOp.Power:
                        GenNumBinOp(node);
                        break;

                    case BinaryOp.Error:
                        PostError(node);
                        break;
                }

                DoConvert(node);
                return false;
            }

            private void GenBoolBinOp(BinaryOpNode node)
            {
                Contracts.AssertValue(node);
                Contracts.Assert(node.Op == BinaryOp.Or || node.Op == BinaryOp.And);
                Contracts.Assert(node.SrcKind == ExprTypeKind.BL);
                Contracts.Assert(node.Left.IsBool);
                Contracts.Assert(node.Right.IsBool);

                // This does naive code gen for short-circuiting binary bool operators.
                // Ideally, this would cooperate with comparisons to produce better code gen.
                // However, this is merely an optimization issue, not correctness, so possibly
                // not worth the additional complexity.

                Label labEnd = _gen.DefineLabel();

                node.Left.Accept(this);
                _gen.Dup();

                if (node.Op == BinaryOp.Or)
                    _gen.Brtrue(labEnd);
                else
                    _gen.Brfalse(labEnd);

                _gen.Pop();
                node.Right.Accept(this);

                _gen.Br(labEnd);
                _gen.MarkLabel(labEnd);
            }

            private void GenNumBinOp(BinaryOpNode node)
            {
                Contracts.AssertValue(node);

                // Note that checking for special known values like NA and identity values
                // is done in the binder and handled in PreVisit(BinaryOpNode).
                node.Left.Accept(this);
                node.Right.Accept(this);

                if (node.SrcKind == ExprTypeKind.I4)
                {
                    Contracts.Assert(node.Left.IsI4);
                    Contracts.Assert(node.Right.IsI4);
                    switch (node.Op)
                    {
                        default:
                            Contracts.Assert(false, "Bad numeric bin op");
                            PostError(node, "Internal error in numeric binary operator");
                            break;

                        case BinaryOp.Add:
                            _gen.Add();
                            break;
                        case BinaryOp.Sub:
                            _gen.Sub();
                            break;
                        case BinaryOp.Mul:
                            _gen.Mul_Ovf();
                            break;
                        case BinaryOp.Div:
                            _gen.Div();
                            break;
                        case BinaryOp.Mod:
                            _gen.Rem();
                            break;
                        case BinaryOp.Power:
                            CallBin<I4>(BuiltinFunctions.Pow);
                            break;
                    }
                }
                else if (node.SrcKind == ExprTypeKind.I8)
                {
                    Contracts.Assert(node.Left.IsI8);
                    Contracts.Assert(node.Right.IsI8);
                    switch (node.Op)
                    {
                        default:
                            Contracts.Assert(false, "Bad numeric bin op");
                            PostError(node, "Internal error in numeric binary operator");
                            break;

                        case BinaryOp.Add:
                            _gen.Add();
                            break;
                        case BinaryOp.Sub:
                            _gen.Sub();
                            break;
                        case BinaryOp.Mul:
                            _gen.Mul();
                            break;
                        case BinaryOp.Div:
                            _gen.Div();
                            break;
                        case BinaryOp.Mod:
                            _gen.Rem();
                            break;
                        case BinaryOp.Power:
                            CallBin<I8>(BuiltinFunctions.Pow);
                            break;
                    }
                }
                else if (node.SrcKind == ExprTypeKind.R4)
                {
                    Contracts.Assert(node.Left.IsR4);
                    Contracts.Assert(node.Right.IsR4);
                    switch (node.Op)
                    {
                        default:
                            Contracts.Assert(false, "Bad numeric bin op");
                            PostError(node, "Internal error in numeric binary operator");
                            break;

                        case BinaryOp.Add:
                            _gen.Add();
                            break;
                        case BinaryOp.Sub:
                            _gen.Sub();
                            break;
                        case BinaryOp.Mul:
                            _gen.Mul();
                            break;
                        case BinaryOp.Div:
                            _gen.Div();
                            break;
                        case BinaryOp.Mod:
                            _gen.Rem();
                            break;
                        case BinaryOp.Power:
                            CallBin<R4>(BuiltinFunctions.Pow);
                            break;
                    }
                }
                else
                {
                    Contracts.Assert(node.SrcKind == ExprTypeKind.R8);
                    Contracts.Assert(node.Left.IsR8);
                    Contracts.Assert(node.Right.IsR8);
                    switch (node.Op)
                    {
                        default:
                            Contracts.Assert(false, "Bad numeric bin op");
                            PostError(node, "Internal error in numeric binary operator");
                            break;

                        case BinaryOp.Add:
                            _gen.Add();
                            break;
                        case BinaryOp.Sub:
                            _gen.Sub();
                            break;
                        case BinaryOp.Mul:
                            _gen.Mul();
                            break;
                        case BinaryOp.Div:
                            _gen.Div();
                            break;
                        case BinaryOp.Mod:
                            _gen.Rem();
                            break;
                        case BinaryOp.Power:
                            CallBin<R8>(Math.Pow);
                            break;
                    }
                }
            }

            private void GenBL(BL value)
            {
                MethodInfo meth;
                if (!value)
                    meth = _methGetFalseBL;
                else
                    meth = _methGetTrueBL;
                _gen.Call(meth);
            }

            private void CallFnc<TSrc, TDst>(Func<TSrc, TDst> fn)
            {
                _gen.Call(fn.GetMethodInfo());
            }

            private void CallFnc<T1, T2, TDst>(Func<T1, T2, TDst> fn)
            {
                _gen.Call(fn.GetMethodInfo());
            }

            private void CallBin<T>(Func<T, T, T> fn)
            {
                _gen.Call(fn.GetMethodInfo());
            }

            private void GenCoalesce(BinaryOpNode node)
            {
                Contracts.AssertValue(node);
                Contracts.Assert(node.Op == BinaryOp.Coalesce);

                // If left is a constant, then the binder should have dealt with it!
                Contracts.Assert(node.Left.ExprValue == null);

                Label labEnd = _gen.DefineLabel();

                // Branch to end if the left operand is NOT NA.
                node.Left.Accept(this);
                GenBrNa(node.Left, labEnd);

                _gen.Pop();
                node.Right.Accept(this);
                _gen.MarkLabel(labEnd);
            }

            public override void PostVisit(BinaryOpNode node)
            {
                Contracts.Assert(false);
            }

            public override bool PreVisit(ConditionalNode node)
            {
                Contracts.AssertValue(node);

                if (TryUseValue(node))
                    return false;

                var cond = (BL?)node.Cond.ExprValue;
                if (cond != null)
                {
                    if (cond.Value)
                        node.Left.Accept(this);
                    else
                        node.Right.Accept(this);
                    goto LDone;
                }

                Label labEnd = _gen.DefineLabel();
                Label labFalse = _gen.DefineLabel();

                node.Cond.Accept(this);
                _gen.Brfalse(labFalse);

                // Left is the "true" branch.
                node.Left.Accept(this);
                _gen.Br(labEnd)
                    .MarkLabel(labFalse);

                node.Right.Accept(this);
                _gen.Br(labEnd);
                _gen.MarkLabel(labEnd);

LDone:
                DoConvert(node);
                return false;
            }

            public override void PostVisit(ConditionalNode node)
            {
                Contracts.Assert(false);
            }

            public override bool PreVisit(CompareNode node)
            {
                Contracts.AssertValue(node);
                Contracts.Assert(node.Operands.Items.Length >= 2);

                if (TryUseValue(node))
                    return false;

                ExprTypeKind kind = node.ArgTypeKind;
                Node[] items = node.Operands.Items;
                if (kind == ExprTypeKind.TX && items.Length == 2)
                {
                    // Two value text comparison is handled by methods.
                    items[0].Accept(this);
                    items[1].Accept(this);
                    switch (node.Op)
                    {
                        default:
                            Contracts.Assert(false, "Bad bool compare op");
                            break;

                        case CompareOp.Equal:
                            CallFnc<TX, TX, BL>(BuiltinFunctions.Equals);
                            break;
                        case CompareOp.NotEqual:
                            CallFnc<TX, TX, BL>(BuiltinFunctions.NotEquals);
                            break;
                    }

                    DoConvert(node);
                    return false;
                }

                Label labEnd = _gen.DefineLabel();
                if (items.Length == 2)
                {
                    // Common case of two operands. Note that the binder should have handled the case when
                    // one or both is a constant NA.

                    ExprNode arg;
                    GenRaw(arg = items[0].AsExpr);
                    Contracts.Assert(arg.ExprType == kind);
                    GenRaw(arg = items[1].AsExpr);
                    Contracts.Assert(arg.ExprType == kind);

                    TokKind tid = node.Operands.Delimiters[0].Kind;
                    Contracts.Assert(tid == node.TidLax || tid == node.TidStrict);
                    var isStrict = tid == node.TidStrict;
                    switch (kind)
                    {
                        case ExprTypeKind.BL:
                            GenCmpBool(node.Op, isStrict);
                            break;
                        case ExprTypeKind.I4:
                        case ExprTypeKind.I8:
                            GenCmpInt(node.Op, isStrict);
                            break;
                        case ExprTypeKind.R4:
                        case ExprTypeKind.R8:
                            GenCmpFloat(node.Op, isStrict);
                            break;

                        default:
                            PostError(node, "Compare codegen for this comparison is NYI");
                            return false;
                    }
                }
                else
                {
                    // For more than two items, we use branching instructions instead of ceq, clt, cgt, etc.
                    Contracts.Assert(items.Length > 2);

                    // Get the comparison generation function and the (raw) local type.
                    Action<CompareOp, bool, Label> fnc;
                    Type typeLoc;
                    switch (kind)
                    {
                        case ExprTypeKind.BL:
                            fnc = GenCmpBool;
                            typeLoc = typeof(byte);
                            break;
                        case ExprTypeKind.TX:
                            fnc = GenCmpText;
                            typeLoc = typeof(TX);
                            break;
                        case ExprTypeKind.I4:
                            fnc = GenCmpInt;
                            typeLoc = typeof(int);
                            break;
                        case ExprTypeKind.I8:
                            fnc = GenCmpInt;
                            typeLoc = typeof(long);
                            break;
                        case ExprTypeKind.R4:
                            fnc = GenCmpFloat;
                            typeLoc = typeof(R4);
                            break;
                        case ExprTypeKind.R8:
                            fnc = GenCmpFloat;
                            typeLoc = typeof(R8);
                            break;

                        default:
                            PostError(node, "Compare codegen for this comparison is NYI");
                            return false;
                    }

                    Label labFalse = _gen.DefineLabel();
                    if (node.Op != CompareOp.NotEqual)
                    {
                        // Note: this loop doesn't work for != so it is handled separately below.
                        ExprNode arg = items[0].AsExpr;
                        Contracts.Assert(arg.ExprType == kind);

                        GenRaw(arg = items[0].AsExpr);
                        Contracts.Assert(arg.ExprType == kind);

                        for (int i = 1; ; i++)
                        {
                            TokKind tid = node.Operands.Delimiters[i - 1].Kind;
                            Contracts.Assert(tid == node.TidLax || tid == node.TidStrict);
                            var isStrict = tid == node.TidStrict;

                            arg = items[i].AsExpr;
                            Contracts.Assert(arg.ExprType == kind);
                            GenRaw(arg);

                            if (i == items.Length - 1)
                            {
                                // Last one.
                                fnc(node.Op, isStrict, labFalse);
                                break;
                            }

                            // We'll need this value again, so stash it in a local.
                            _gen.Dup();
                            using (var local = _meth.AcquireTemporary(typeLoc))
                            {
                                _gen.Stloc(local.Local);
                                fnc(node.Op, isStrict, labFalse);
                                _gen.Ldloc(local.Local);
                            }
                        }
                    }
                    else
                    {
                        // NotEqual is special - it means that the values are all distinct, so comparing adjacent
                        // items is not enough.
                        Contracts.Assert(node.Op == CompareOp.NotEqual && items.Length > 2);

                        // We need a local for each item.
                        var locals = new MethodGenerator.Temporary[items.Length];
                        for (int i = 0; i < locals.Length; i++)
                            locals[i] = _meth.AcquireTemporary(typeLoc);
                        try
                        {
                            ExprNode arg = items[0].AsExpr;
                            Contracts.Assert(arg.ExprType == kind);

                            GenRaw(arg);
                            _gen.Stloc(locals[0].Local);

                            for (int i = 1; i < items.Length; i++)
                            {
                                // Need to evaluate the expression and store it in the local.
                                arg = items[i].AsExpr;
                                Contracts.Assert(arg.ExprType == kind);
                                GenRaw(arg);
                                _gen.Stloc(locals[i].Local);

                                for (int j = 0; j < i; j++)
                                {
                                    _gen.Ldloc(locals[j].Local)
                                        .Ldloc(locals[i].Local);
                                    fnc(node.Op, true, labFalse);
                                }
                            }
                        }
                        finally
                        {
                            for (int i = locals.Length; --i >= 0;)
                                locals[i].Dispose();
                        }
                    }

                    _gen.Call(_methGetTrueBL)
                        .Br(labEnd);

                    _gen.MarkLabel(labFalse);
                    _gen.Call(_methGetFalseBL)
                        .Br(labEnd);
                }

                _gen.MarkLabel(labEnd);

                DoConvert(node);
                return false;
            }

            /// <summary>
            /// Get the raw bits from an expression node. If the node is constant, this avoids the
            /// silly "convert to dv type" followed by "extract raw bits". Returns whether the expression
            /// is a constant NA, with null meaning "don't know".
            /// </summary>
            private void GenRaw(ExprNode node)
            {
                Contracts.AssertValue(node);

                var val = node.ExprValue;
                if (val != null)
                {
                    switch (node.ExprType)
                    {
                        case ExprTypeKind.BL:
                            {
                                var x = (BL)val;
                                _gen.Ldc_I4(x ? 1 : 0);
                                return;
                            }
                        case ExprTypeKind.I4:
                            {
                                var x = (I4)val;
                                _gen.Ldc_I4(x);
                                return;
                            }
                        case ExprTypeKind.I8:
                            {
                                var x = (I8)val;
                                _gen.Ldc_I8(x);
                                return;
                            }
                        case ExprTypeKind.R4:
                            {
                                var x = (R4)val;
                                _gen.Ldc_R4(x);
                                return;
                            }
                        case ExprTypeKind.R8:
                            {
                                var x = (R8)val;
                                _gen.Ldc_R8(x);
                                return;
                            }
                        case ExprTypeKind.TX:
                            {
                                var x = (TX)val;
                                _gen.Ldstr(x.ToString());
                                CallFnc<string, TX>(Exec.ToTX);
                                return;
                            }
                    }
                }

                node.Accept(this);
            }

            /// <summary>
            /// Generate code to branch to labNa if the top stack element is NA.
            /// Note that this leaves the element on the stack (duplicates before comparing).
            /// If rev is true, this branches when NOT NA.
            /// </summary>
            private void GenBrNa(ExprNode node, Label labNa, bool dup = true)
            {
                GenBrNaCore(node, node.ExprType, labNa, dup);
            }

            /// <summary>
            /// Generate code to branch to labNa if the top stack element is NA.
            /// If dup is true, this leaves the element on the stack (duplicates before comparing).
            /// If rev is true, this branches when NOT NA.
            /// </summary>
            private void GenBrNaCore(ExprNode node, ExprTypeKind kind, Label labNa, bool dup)
            {
                if (dup)
                    _gen.Dup();

                switch (kind)
                {
                    case ExprTypeKind.R4:
                    case ExprTypeKind.R8:
                        // Any value that is not equal to itself is an NA.
                        _gen.Dup();
                        _gen.Beq(labNa);
                        break;
                    case ExprTypeKind.Error:
                    case ExprTypeKind.None:
                        Contracts.Assert(false, "Bad expr kind in GenBrNa");
                        PostError(node, "Internal error in GenBrNa");
                        break;
                }
            }

            /// <summary>
            /// Generate a bool from comparing the raw bits. The values are guaranteed to not be NA.
            /// </summary>
            private void GenCmpBool(CompareOp op, bool isStrict)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad bool compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Ceq();
                        break;
                    case CompareOp.NotEqual:
                        _gen.Xor();
                        break;
                }
            }

            /// <summary>
            /// Generate a bool from comparing the raw bits. The values are guaranteed to not be NA.
            /// </summary>
            private void GenCmpInt(CompareOp op, bool isStrict)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Ceq();
                        break;
                    case CompareOp.NotEqual:
                        _gen.Ceq().Ldc_I4(0).Ceq();
                        break;
                    case CompareOp.DecrChain:
                        if (isStrict)
                            _gen.Cgt();
                        else
                            _gen.Clt().Ldc_I4(0).Ceq();
                        break;
                    case CompareOp.IncrChain:
                        if (isStrict)
                            _gen.Clt();
                        else
                            _gen.Cgt().Ldc_I4(0).Ceq();
                        break;
                }
            }

            /// <summary>
            /// Generate a bool from comparing the raw bits. The values are guaranteed to not be NA.
            /// </summary>
            private void GenCmpFloat(CompareOp op, bool isStrict)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Ceq();
                        break;
                    case CompareOp.NotEqual:
                        _gen.Ceq().Ldc_I4(0).Ceq();
                        break;
                    case CompareOp.DecrChain:
                        if (isStrict)
                            _gen.Cgt();
                        else
                            _gen.Clt_Un().Ldc_I4(0).Ceq();
                        break;
                    case CompareOp.IncrChain:
                        if (isStrict)
                            _gen.Clt();
                        else
                            _gen.Cgt_Un().Ldc_I4(0).Ceq();
                        break;
                }
            }

            private void GenCmpBool(CompareOp op, bool isStrict, Label labFalse)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad bool compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Bne_Un(labFalse);
                        break;
                    case CompareOp.NotEqual:
                        _gen.Beq(labFalse);
                        break;
                }
            }

            private void GenCmpText(CompareOp op, bool isStrict, Label labFalse)
            {
                // Note that NA values don't come through here, so we don't need NA propagating equality comparison.
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad bool compare op");
                        break;

                    case CompareOp.Equal:
                        CallFnc<TX, TX, bool>(BuiltinFunctions.Equals);
                        _gen.Brfalse(labFalse);
                        break;
                    case CompareOp.NotEqual:
                        CallFnc<TX, TX, bool>(BuiltinFunctions.Equals);
                        _gen.Brtrue(labFalse);
                        break;
                }
            }

            private void GenCmpInt(CompareOp op, bool isStrict, Label labFalse)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Bne_Un(labFalse);
                        break;
                    case CompareOp.NotEqual:
                        _gen.Beq(labFalse);
                        break;
                    case CompareOp.DecrChain:
                        if (isStrict)
                            _gen.Ble(labFalse);
                        else
                            _gen.Blt(labFalse);
                        break;
                    case CompareOp.IncrChain:
                        if (isStrict)
                            _gen.Bge(labFalse);
                        else
                            _gen.Bgt(labFalse);
                        break;
                }
            }

            private void GenCmpFloat(CompareOp op, bool isStrict, Label labFalse)
            {
                switch (op)
                {
                    default:
                        Contracts.Assert(false, "Bad compare op");
                        break;

                    case CompareOp.Equal:
                        _gen.Bne_Un(labFalse);
                        break;
                    case CompareOp.NotEqual:
                        _gen.Beq(labFalse);
                        break;
                    case CompareOp.DecrChain:
                        if (isStrict)
                            _gen.Ble_Un(labFalse);
                        else
                            _gen.Blt_Un(labFalse);
                        break;
                    case CompareOp.IncrChain:
                        if (isStrict)
                            _gen.Bge_Un(labFalse);
                        else
                            _gen.Bgt_Un(labFalse);
                        break;
                }
            }

            public override void PostVisit(CompareNode node)
            {
                Contracts.Assert(false);
            }

            public override bool PreVisit(CallNode node)
            {
                Contracts.AssertValue(node);

                if (TryUseValue(node))
                    return false;

                if (node.Method == null)
                {
                    Contracts.Assert(false, "Bad function");
                    PostError(node, "Internal error: unknown function: '{0}'", node.Head.Value);
                    return false;
                }

                var meth = node.Method;
                var ps = meth.GetParameters();
                Type type;
                if (Utils.Size(ps) > 0 && (type = ps[ps.Length - 1].ParameterType).IsArray)
                {
                    // Variable case, so can't be identity.
                    Contracts.Assert(node.Method.ReturnType != typeof(void));

                    // Get the item type of the array.
                    type = type.GetElementType();

                    var args = node.Args.Items;
                    int head = ps.Length - 1;
                    int tail = node.Args.Items.Length - head;
                    Contracts.Assert(tail >= 0);

                    // Generate the "head" args.
                    for (int i = 0; i < head; i++)
                        args[i].Accept(this);

                    // Bundle the "tail" args into an array.
                    _gen.Ldc_I4(tail)
                        .Newarr(type);
                    for (int i = 0; i < tail; i++)
                    {
                        _gen.Dup()
                            .Ldc_I4(i);
                        args[head + i].Accept(this);
                        _gen.Stelem(type);
                    }

                    // Make the call.
                    _gen.Call(node.Method);
                }
                else
                {
                    Contracts.Assert(Utils.Size(ps) == node.Args.Items.Length);
                    node.Args.Accept(this);

                    // An identity function is marked with a void return type.
                    if (node.Method.ReturnType != typeof(void))
                        _gen.Call(node.Method);
                    else
                        Contracts.Assert(node.Args.Items.Length == 1);
                }

                DoConvert(node);
                return false;
            }

            public override void PostVisit(CallNode node)
            {
                Contracts.Assert(false);
            }

            public override void PostVisit(ListNode node)
            {
                Contracts.AssertValue(node);
            }

            public override bool PreVisit(WithNode node)
            {
                Contracts.AssertValue(node);

                var local = node.Local;
                Contracts.Assert(local.Index == -1);
                Contracts.Assert(local.UseCount >= 0);

                if (local.Value.ExprValue != null || local.UseCount <= 1)
                {
                    // In this case, simply inline the code generation, no need
                    // to cache the value in an IL local.
                    node.Body.Accept(this);
                    Contracts.Assert(local.Index == -1);
                }
                else
                {
                    // REVIEW: What's a reasonable value? This allows binary uses of 7 locals.
                    // This should cover most cases, but allows a rather large bloat factor.
                    const int maxTotalUse = 128;

                    // This case uses a cache value. When lazy, it also keeps a bool flag indicating
                    // whether the value has been computed and stored in the cache yet.
                    int index = _cacheWith.Count;

                    // Lazy can bloat code gen exponentially. This test decides whether to be lazy for this
                    // particular local, based on its use count and nesting. This assumes the worst case,
                    // that each lazy value is used by the next lazy value the full UseCount number of times.
                    // REVIEW: We should try to do better at some point.... Strictness analysis would
                    // solve this, but is non-trivial to implement.
                    bool lazy = true;
                    long totalUse = local.UseCount;
                    if (totalUse > maxTotalUse)
                        lazy = false;
                    else
                    {
                        for (int i = index; --i >= 0;)
                        {
                            var item = _cacheWith[i];
                            Contracts.Assert(item.Node.UseCount >= 2);
                            if (item.Flag == null)
                                continue;
                            totalUse *= item.Node.UseCount;
                            if (totalUse > maxTotalUse)
                            {
                                lazy = false;
                                break;
                            }
                        }
                    }

                    // This risks code gen
                    // bloat but avoids unnecessary computation. Perhaps we should determine whether the
                    // value is always needed. However, this can be quite complicated, requiring flow
                    // analysis through all expression kinds.

                    // REVIEW: Should we always make the code generation lazy? This risks code gen
                    // bloat but avoids unnecessary computation. Perhaps we should determine whether the
                    // value is always needed. However, this can be quite complicated, requiring flow
                    // analysis through all expression kinds.
                    using (var value = _meth.AcquireTemporary(ExprNode.ToSysType(local.Value.ExprType)))
                    using (var flag = lazy ? _meth.AcquireTemporary(typeof(bool)) : default(MethodGenerator.Temporary))
                    {
                        LocalBuilder flagBldr = flag.Local;
                        Contracts.Assert((flagBldr != null) == lazy);

                        if (lazy)
                        {
                            _gen
                                .Ldc_I4(0)
                                .Stloc(flagBldr);
                        }
                        else
                        {
                            local.Value.Accept(this);
                            _gen.Stloc(value.Local);
                        }

                        // Activate the cache item.
                        var cache = new CachedWithLocal(local, value.Local, flag.Local);
                        _cacheWith.Add(cache);
                        Contracts.Assert(_cacheWith.Count == index + 1);

                        // Generate the code for the body.
                        local.Index = index;
                        node.Body.Accept(this);
                        Contracts.Assert(local.Index == index);
                        local.Index = -1;

                        // Remove the cache locals.
                        Contracts.Assert(_cacheWith.Count == index + 1);
                        Contracts.Assert(_cacheWith[index] == cache);
                        _cacheWith.RemoveAt(index);
                    }
                }

#if DEBUG
                System.Diagnostics.Debug.WriteLine("Generated code '{0}' times for '{1}'", local.GenCount, local);
#endif
                return false;
            }

            public override void PostVisit(WithNode node)
            {
                Contracts.Assert(false);
            }

            public override bool PreVisit(WithLocalNode node)
            {
                Contracts.Assert(false);
                return false;
            }

            public override void PostVisit(WithLocalNode node)
            {
                Contracts.Assert(false);
            }
        }
    }
}
