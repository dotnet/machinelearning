// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.ML.Data;
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

    internal sealed partial class LambdaBinder : NodeVisitor
    {
        private readonly IHost _host;
        // The stack of active with nodes.
        private readonly List<WithNode> _rgwith;

        private List<Error> _errors;
        private LambdaNode _lambda;

        private readonly IFunctionProvider[] _providers;
        private readonly Action<string> _printError;

        private LambdaBinder(IHostEnvironment env, Action<string> printError)
        {
            _host = env.Register("LambdaBinder");
            _printError = printError;
            _rgwith = new List<WithNode>();
            _providers = env.ComponentCatalog.GetAllDerivedClasses(typeof(IFunctionProvider), typeof(SignatureFunctionProvider))
                .Select(info => info.CreateInstance<IFunctionProvider>(_host))
                .Prepend(BuiltinFunctions.Instance)
                .ToArray();
        }

        /// <summary>
        /// Run Lambda binder on LambdaNode and populate Expr values.
        /// The errors contain list of user errors that occurred during binding.
        /// The printError delegate is only used for reporting issues with function provider implementations, which are programmer errors.
        /// In particular, it is NOT used to report user errors in the lambda expression.
        /// </summary>
        public static void Run(IHostEnvironment env, ref List<Error> errors, LambdaNode node, Action<string> printError)
        {
            Contracts.AssertValue(env);
            env.AssertValueOrNull(errors);
            env.AssertValue(node);
            env.AssertValue(printError);

            var binder = new LambdaBinder(env, printError);
            binder._errors = errors;
            node.Accept(binder);
            env.Assert(binder._rgwith.Count == 0);

            var expr = node.Expr;
            switch (expr.ExprType)
            {
                case ExprTypeKind.BL:
                    node.ResultType = BooleanDataViewType.Instance;
                    break;
                case ExprTypeKind.I4:
                    node.ResultType = NumberDataViewType.Int32;
                    break;
                case ExprTypeKind.I8:
                    node.ResultType = NumberDataViewType.Int64;
                    break;
                case ExprTypeKind.R4:
                    node.ResultType = NumberDataViewType.Single;
                    break;
                case ExprTypeKind.R8:
                    node.ResultType = NumberDataViewType.Double;
                    break;
                case ExprTypeKind.TX:
                    node.ResultType = TextDataViewType.Instance;
                    break;
                default:
                    if (!binder.HasErrors)
                        binder.PostError(expr, "Invalid result type");
                    break;
            }

            errors = binder._errors;
        }

        private bool HasErrors
        {
            get { return Utils.Size(_errors) > 0; }
        }

        private void PostError(Node node, string msg)
        {
            Utils.Add(ref _errors, new Error(node.Token, msg));
        }

        private void PostError(Node node, string msg, params object[] args)
        {
            Utils.Add(ref _errors, new Error(node.Token, string.Format(msg, args)));
        }

        public override void Visit(BoolLitNode node)
        {
            _host.AssertValue(node);
            _host.Assert(node.IsBool);
            _host.AssertValue(node.ExprValue);
        }

        public override void Visit(StrLitNode node)
        {
            _host.AssertValue(node);
            _host.Assert(node.IsTX);
            _host.AssertValue(node.ExprValue);
        }

        public override void Visit(NumLitNode node)
        {
            _host.AssertValue(node);
            _host.Assert(node.IsNumber || node.IsError);
            _host.Assert((node.ExprValue == null) == node.IsError);

            if (node.IsError)
                PostError(node, "Overflow");
        }

        public override void Visit(NameNode node)
        {
        }

        public override void Visit(IdentNode node)
        {
            _host.AssertValue(node);

            // If the IdentNode didn't actually have an IdentToken, just bag out.
            if (node.IsMissing)
            {
                _host.Assert(HasErrors);
                node.SetType(ExprTypeKind.Error);
                return;
            }

            // Look for "with" locals.
            string name = node.Value;
            for (int i = _rgwith.Count; --i >= 0;)
            {
                var with = _rgwith[i];
                if (name == with.Local.Name)
                {
                    node.Referent = with.Local;
                    node.SetValue(with.Local.Value);
                    // REVIEW: Note that some uses might get pruned, but this gives us
                    // an upper bound on the time of places in the code where this value is needed.
                    with.Local.UseCount++;
                    return;
                }
            }

            // Look for parameters.
            ParamNode param;
            if (_lambda != null && (param = _lambda.FindParam(node.Value)) != null)
            {
                node.Referent = param;
                node.SetType(param.ExprType);
                return;
            }

            PostError(node, "Unresolved identifier '{0}'", node.Value);
            node.SetType(ExprTypeKind.Error);
        }

        public override void Visit(ParamNode node)
        {
            _host.AssertValue(node);
            _host.Assert(node.ExprType != 0);
        }

        public override bool PreVisit(LambdaNode node)
        {
            _host.AssertValue(node);
            _host.Assert(_lambda == null, "Can't support nested lambdas");

            _lambda = node;

            node.Expr.Accept(this);

            _host.Assert(_lambda == node);
            _lambda = null;

            return false;
        }

        public override void PostVisit(LambdaNode node)
        {
            _host.Assert(false);
        }

        private string GetStr(ExprTypeKind kind)
        {
            switch (kind)
            {
                case ExprTypeKind.BL:
                    return "boolean";
                case ExprTypeKind.R4:
                case ExprTypeKind.R8:
                    return "numeric";
                case ExprTypeKind.I4:
                case ExprTypeKind.I8:
                    return "integer";
                case ExprTypeKind.TX:
                    return "text";
            }

            return null;
        }

        private void BadNum(ExprNode arg)
        {
            if (!arg.IsError)
                PostError(arg, "Invalid numeric operand");
            _host.Assert(HasErrors);
        }

        private void BadNum(ExprNode node, ExprNode arg)
        {
            BadNum(arg);
            _host.Assert(HasErrors);
            node.SetType(ExprTypeKind.Error);
        }

        private void BadText(ExprNode arg)
        {
            if (!arg.IsError)
                PostError(arg, "Invalid text operand");
            _host.Assert(HasErrors);
        }

        private void BadArg(ExprNode arg, ExprTypeKind kind)
        {
            if (!arg.IsError)
            {
                var str = GetStr(kind);
                if (str != null)
                    PostError(arg, "Invalid {0} operand", str);
                else
                    PostError(arg, "Invalid operand");
            }
            _host.Assert(HasErrors);
        }

        public override void PostVisit(UnaryOpNode node)
        {
            _host.AssertValue(node);
            var arg = node.Arg;
            switch (node.Op)
            {
                case UnaryOp.Minus:
                    switch (arg.ExprType)
                    {
                        default:
                            BadNum(node, arg);
                            break;
                        case ExprTypeKind.I4:
                            node.SetValue(-(I4?)arg.ExprValue);
                            break;
                        case ExprTypeKind.I8:
                            node.SetValue(-(I8?)arg.ExprValue);
                            break;
                        case ExprTypeKind.R4:
                            node.SetValue(-(R4?)arg.ExprValue);
                            break;
                        case ExprTypeKind.R8:
                            node.SetValue(-(R8?)arg.ExprValue);
                            break;
                    }
                    break;

                case UnaryOp.Not:
                    BL? bl = GetBoolOp(node.Arg);
                    if (bl != null)
                        node.SetValue(!bl.Value);
                    else
                        node.SetValue(bl);
                    break;

                default:
                    _host.Assert(false);
                    PostError(node, "Unknown unary operator");
                    node.SetType(ExprTypeKind.Error);
                    break;
            }
        }

        private BL? GetBoolOp(ExprNode arg)
        {
            _host.AssertValue(arg);
            if (arg.IsBool)
                return (BL?)arg.ExprValue;
            BadArg(arg, ExprTypeKind.BL);
            return null;
        }

        public override void PostVisit(BinaryOpNode node)
        {
            _host.AssertValue(node);

            // REVIEW: We should really use the standard function overload resolution
            // mechanism that CallNode binding uses. That would ensure that our type promotion
            // and resolution mechanisms are consistent.
            switch (node.Op)
            {
                case BinaryOp.Coalesce:
                    if (!node.Left.IsRx)
                    {
                        BadArg(node, node.Left.ExprType);
                        node.SetType(ExprTypeKind.Error);
                    }
                    else // Default to numeric.
                        ApplyNumericBinOp(node);
                    break;

                case BinaryOp.Or:
                case BinaryOp.And:
                    ApplyBoolBinOp(node);
                    break;

                case BinaryOp.Add:
                case BinaryOp.Sub:
                case BinaryOp.Mul:
                case BinaryOp.Div:
                case BinaryOp.Mod:
                case BinaryOp.Power:
                    ApplyNumericBinOp(node);
                    break;

                case BinaryOp.Error:
                    _host.Assert(HasErrors);
                    node.SetType(ExprTypeKind.Error);
                    break;

                default:
                    _host.Assert(false);
                    PostError(node, "Unknown binary operator");
                    node.SetType(ExprTypeKind.Error);
                    break;
            }
        }

        private void ApplyBoolBinOp(BinaryOpNode node)
        {
            _host.AssertValue(node);
            _host.Assert(node.Op == BinaryOp.And || node.Op == BinaryOp.Or || node.Op == BinaryOp.Coalesce);

            node.SetType(ExprTypeKind.BL);

            BL? v1 = GetBoolOp(node.Left);
            BL? v2 = GetBoolOp(node.Right);
            switch (node.Op)
            {
                case BinaryOp.Or:
                    if (v1 != null && v2 != null)
                        node.SetValue(v1.Value || v2.Value);
                    else if (v1 != null && v1.Value || v2 != null && v2.Value)
                        node.SetValue(true);
                    else if (v1 != null && !v1.Value)
                        node.ReduceToRight = true;
                    else if (v2 != null && !v2.Value)
                        node.ReduceToLeft = true;
                    break;

                case BinaryOp.And:
                    if (v1 != null && v2 != null)
                        node.SetValue(v1.Value && v2.Value);
                    else if (v1 != null && !v1.Value || v2 != null && !v2.Value)
                        node.SetValue(false);
                    else if (v1 != null && v1.Value)
                        node.ReduceToRight = true;
                    else if (v2 != null && v2.Value)
                        node.ReduceToLeft = true;
                    break;

                case BinaryOp.Coalesce:
                    if (v1 != null)
                        node.SetValue(v1);
                    break;
            }

            _host.Assert(node.IsBool);
        }

        /// <summary>
        /// Reconcile the types of the two ExprNodes. Favor numeric types in cases
        /// where the types can't be reconciled. This does not guarantee that
        /// the resulting kind is numeric, eg, if both a and b are of type Text, it
        /// simply sets kind to Text.
        /// </summary>
        private void ReconcileNumericTypes(ExprNode a, ExprNode b, out ExprTypeKind kind)
        {
            _host.AssertValue(a);
            _host.AssertValue(b);

            // REVIEW: Consider converting I4 + R4 to R8, unless the I4
            // is a constant known to not lose precision when converted to R4.
            if (!CanPromote(false, a.ExprType, b.ExprType, out kind))
            {
                // If either is numeric, use that numeric type.
                if (a.IsNumber)
                    kind = a.ExprType;
                else if (b.IsNumber)
                    kind = b.ExprType;
                else // Default to Float (for error reporting).
                    kind = ExprTypeKind.Float;
                _host.Assert(MapKindToIndex(kind) >= 0);
            }
        }

        private void ApplyNumericBinOp(BinaryOpNode node)
        {
            _host.AssertValue(node);

            var left = node.Left;
            var right = node.Right;
            ExprTypeKind kind;
            ReconcileNumericTypes(left, right, out kind);

            // REVIEW: Should we prohibit constant evaluations that produce NA?
            switch (kind)
            {
                default:
                    // Default to Float (for error reporting).
                    goto case ExprTypeKind.Float;

                case ExprTypeKind.I4:
                    {
                        node.SetType(ExprTypeKind.I4);
                        I4? v1;
                        I4? v2;
                        // Boiler plate below here...
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        else if (!f2)
                            BadNum(right);
                        else
                            ReduceBinOp(node, v1, v2);
                    }
                    break;
                case ExprTypeKind.I8:
                    {
                        node.SetType(ExprTypeKind.I8);
                        I8? v1;
                        I8? v2;
                        // Boiler plate below here...
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        else if (!f2)
                            BadNum(right);
                        else
                            ReduceBinOp(node, v1, v2);
                    }
                    break;
                case ExprTypeKind.R4:
                    {
                        node.SetType(ExprTypeKind.R4);
                        R4? v1;
                        R4? v2;
                        // Boiler plate below here...
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        else if (!f2)
                            BadNum(right);
                        else
                            ReduceBinOp(node, v1, v2);
                    }
                    break;
                case ExprTypeKind.R8:
                    {
                        node.SetType(ExprTypeKind.R8);
                        R8? v1;
                        R8? v2;
                        // Boiler plate below here...
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        else if (!f2)
                            BadNum(right);
                        else
                            ReduceBinOp(node, v1, v2);
                    }
                    break;
            }
        }

        #region ReduceBinOp

        // The I4 and I8 methods are identical, as are the R4 and R8 methods.
        private void ReduceBinOp(BinaryOpNode node, I4? a, I4? b)
        {
            if (a != null && b != null)
                node.SetValue(BinOp(node, a.Value, b.Value));
            else if (a != null)
            {
                // Special reductions when only the left value is known.
                var v = a.Value;
                switch (node.Op)
                {
                    case BinaryOp.Add:
                        if (v == 0)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Mul:
                        if (v == 1)
                            node.ReduceToRight = true;
                        break;
                }
            }
            else if (b != null)
            {
                // Special reductions when only the right value is known.
                var v = b.Value;
                switch (node.Op)
                {
                    case BinaryOp.Add:
                        if (v == 0)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Mul:
                        if (v == 1)
                            node.ReduceToLeft = true;
                        break;
                }
            }
        }

        private void ReduceBinOp(BinaryOpNode node, I8? a, I8? b)
        {
            if (a != null && b != null)
                node.SetValue(BinOp(node, a.Value, b.Value));
            else if (a != null)
            {
                // Special reductions when only the left value is known.
                var v = a.Value;
                switch (node.Op)
                {
                    case BinaryOp.Add:
                        if (v == 0)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Mul:
                        if (v == 1)
                            node.ReduceToRight = true;
                        break;
                }
            }
            else if (b != null)
            {
                // Special reductions when only the right value is known.
                var v = b.Value;
                switch (node.Op)
                {
                    case BinaryOp.Add:
                        if (v == 0)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Mul:
                        if (v == 1)
                            node.ReduceToLeft = true;
                        break;
                }
            }
        }

        private void ReduceBinOp(BinaryOpNode node, R4? a, R4? b)
        {
            if (a != null && b != null)
                node.SetValue(BinOp(node, a.Value, b.Value));
            else if (a != null)
            {
                // Special reductions when only the left value is known.
                var v = a.Value;
                switch (node.Op)
                {
                    case BinaryOp.Coalesce:
                        if (!R4.IsNaN(v))
                            node.SetValue(v);
                        else
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Add:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 0)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Mul:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 1)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Sub:
                    case BinaryOp.Div:
                    case BinaryOp.Mod:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        break;
                }
            }
            else if (b != null)
            {
                // Special reductions when only the right value is known.
                var v = b.Value;
                switch (node.Op)
                {
                    case BinaryOp.Coalesce:
                        if (R4.IsNaN(v))
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Add:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 0)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Mul:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 1)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Sub:
                    case BinaryOp.Div:
                    case BinaryOp.Mod:
                        if (R4.IsNaN(v))
                            node.SetValue(v);
                        break;
                }
            }
        }

        private void ReduceBinOp(BinaryOpNode node, R8? a, R8? b)
        {
            if (a != null && b != null)
                node.SetValue(BinOp(node, a.Value, b.Value));
            else if (a != null)
            {
                // Special reductions when only the left value is known.
                var v = a.Value;
                switch (node.Op)
                {
                    case BinaryOp.Coalesce:
                        if (!R8.IsNaN(v))
                            node.SetValue(v);
                        else
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Add:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 0)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Mul:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 1)
                            node.ReduceToRight = true;
                        break;
                    case BinaryOp.Sub:
                    case BinaryOp.Div:
                    case BinaryOp.Mod:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        break;
                }
            }
            else if (b != null)
            {
                // Special reductions when only the right value is known.
                var v = b.Value;
                switch (node.Op)
                {
                    case BinaryOp.Coalesce:
                        if (R8.IsNaN(v))
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Add:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 0)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Mul:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        else if (v == 1)
                            node.ReduceToLeft = true;
                        break;
                    case BinaryOp.Sub:
                    case BinaryOp.Div:
                    case BinaryOp.Mod:
                        if (R8.IsNaN(v))
                            node.SetValue(v);
                        break;
                }
            }
        }

        #endregion ReduceBinOp

        #region BinOp

        private I4 BinOp(BinaryOpNode node, I4 v1, I4 v2)
        {
            switch (node.Op)
            {
                case BinaryOp.Add:
                    return v1 + v2;
                case BinaryOp.Sub:
                    return v1 - v2;
                case BinaryOp.Mul:
                    return v1 * v2;
                case BinaryOp.Div:
                    return v1 / v2;
                case BinaryOp.Mod:
                    return v1 % v2;
                case BinaryOp.Power:
                    return BuiltinFunctions.Pow(v1, v2);
                default:
                    _host.Assert(false);
                    throw Contracts.Except();
            }
        }

        private I8 BinOp(BinaryOpNode node, I8 v1, I8 v2)
        {
            switch (node.Op)
            {
                case BinaryOp.Add:
                    return v1 + v2;
                case BinaryOp.Sub:
                    return v1 - v2;
                case BinaryOp.Mul:
                    return v1 * v2;
                case BinaryOp.Div:
                    return v1 / v2;
                case BinaryOp.Mod:
                    return v1 % v2;
                case BinaryOp.Power:
                    return BuiltinFunctions.Pow(v1, v2);
                default:
                    _host.Assert(false);
                    throw Contracts.Except();
            }
        }

        private R4 BinOp(BinaryOpNode node, R4 v1, R4 v2)
        {
            switch (node.Op)
            {
                case BinaryOp.Coalesce:
                    return !R4.IsNaN(v1) ? v1 : v2;
                case BinaryOp.Add:
                    return v1 + v2;
                case BinaryOp.Sub:
                    return v1 - v2;
                case BinaryOp.Mul:
                    return v1 * v2;
                case BinaryOp.Div:
                    return v1 / v2;
                case BinaryOp.Mod:
                    return v1 % v2;
                case BinaryOp.Power:
                    return BuiltinFunctions.Pow(v1, v2);
                default:
                    _host.Assert(false);
                    return R4.NaN;
            }
        }

        private R8 BinOp(BinaryOpNode node, R8 v1, R8 v2)
        {
            switch (node.Op)
            {
                case BinaryOp.Coalesce:
                    return !R8.IsNaN(v1) ? v1 : v2;
                case BinaryOp.Add:
                    return v1 + v2;
                case BinaryOp.Sub:
                    return v1 - v2;
                case BinaryOp.Mul:
                    return v1 * v2;
                case BinaryOp.Div:
                    return v1 / v2;
                case BinaryOp.Mod:
                    return v1 % v2;
                case BinaryOp.Power:
                    return Math.Pow(v1, v2);
                default:
                    _host.Assert(false);
                    return R8.NaN;
            }
        }
        #endregion BinOp

        public override void PostVisit(ConditionalNode node)
        {
            _host.AssertValue(node);

            BL? cond = GetBoolOp(node.Cond);

            var left = node.Left;
            var right = node.Right;
            ExprTypeKind kind;
            if (!CanPromote(false, left.ExprType, right.ExprType, out kind))
            {
                // If either is numeric, use that numeric type. Otherwise, use the first
                // that isn't error or none.
                if (left.IsNumber)
                    kind = left.ExprType;
                else if (right.IsNumber)
                    kind = right.ExprType;
                else if (!left.IsError && !left.IsNone)
                    kind = left.ExprType;
                else if (!right.IsError && !right.IsNone)
                    kind = right.ExprType;
                else
                    kind = ExprTypeKind.None;
            }

            switch (kind)
            {
                default:
                    PostError(node, "Invalid conditional expression");
                    node.SetType(ExprTypeKind.Error);
                    break;

                case ExprTypeKind.BL:
                    {
                        node.SetType(ExprTypeKind.BL);
                        BL? v1 = GetBoolOp(node.Left);
                        BL? v2 = GetBoolOp(node.Right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
                case ExprTypeKind.I4:
                    {
                        node.SetType(ExprTypeKind.I4);
                        I4? v1;
                        I4? v2;
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        if (!f2)
                            BadNum(right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
                case ExprTypeKind.I8:
                    {
                        node.SetType(ExprTypeKind.I8);
                        I8? v1;
                        I8? v2;
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        if (!f2)
                            BadNum(right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
                case ExprTypeKind.R4:
                    {
                        node.SetType(ExprTypeKind.R4);
                        R4? v1;
                        R4? v2;
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        if (!f2)
                            BadNum(right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
                case ExprTypeKind.R8:
                    {
                        node.SetType(ExprTypeKind.R8);
                        R8? v1;
                        R8? v2;
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadNum(left);
                        if (!f2)
                            BadNum(right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
                case ExprTypeKind.TX:
                    {
                        node.SetType(ExprTypeKind.TX);
                        TX? v1;
                        TX? v2;
                        bool f1 = left.TryGet(out v1);
                        bool f2 = right.TryGet(out v2);
                        if (!f1)
                            BadText(left);
                        if (!f2)
                            BadText(right);
                        if (cond != null)
                        {
                            if (cond.Value)
                                node.SetValue(v1);
                            else
                                node.SetValue(v2);
                        }
                    }
                    break;
            }
        }

        public override void PostVisit(CompareNode node)
        {
            _host.AssertValue(node);

            TokKind tidLax = node.TidLax;
            TokKind tidStrict = node.TidStrict;
            ExprTypeKind kind = ExprTypeKind.None;

            // First validate the types.
            ExprNode arg;
            bool hasErrors = false;
            var items = node.Operands.Items;
            for (int i = 0; i < items.Length; i++)
            {
                arg = items[i].AsExpr;
                if (!ValidateType(arg, ref kind))
                {
                    BadArg(arg, kind);
                    hasErrors = true;
                }
            }

            // Set the arg type and the type of this node.
            node.ArgTypeKind = kind;
            node.SetType(ExprTypeKind.BL);

            if (hasErrors)
            {
                _host.Assert(HasErrors);
                return;
            }

            // Find the number of initial constant inputs in "lim" and convert the args to "kind".
            int lim = items.Length;
            int count = lim;
            for (int i = 0; i < count; i++)
            {
                arg = items[i].AsExpr;
                arg.Convert(kind);
                if (i < lim && arg.ExprValue == null)
                    lim = i;
            }

            // Now try to compute the value.
            int ifn = (int)kind;
            if (ifn >= _fnEqual.Length || ifn < 0)
            {
                _host.Assert(false);
                PostError(node, "Internal error in CompareNode");
                return;
            }

            Cmp cmpLax;
            Cmp cmpStrict;
            switch (node.Op)
            {
                case CompareOp.DecrChain:
                    cmpLax = _fnGreaterEqual[ifn];
                    cmpStrict = _fnGreater[ifn];
                    break;
                case CompareOp.IncrChain:
                    cmpLax = _fnLessEqual[ifn];
                    cmpStrict = _fnLess[ifn];
                    break;
                case CompareOp.Equal:
                    cmpLax = _fnEqual[ifn];
                    cmpStrict = cmpLax;
                    break;
                case CompareOp.NotEqual:
                    cmpLax = _fnNotEqual[ifn];
                    cmpStrict = cmpLax;
                    break;
                default:
                    _host.Assert(false);
                    return;
            }

            _host.Assert((cmpLax == null) == (cmpStrict == null));
            if (cmpLax == null)
            {
                PostError(node, "Bad operands for comparison");
                return;
            }

            // If one of the first two operands is NA, the result is NA, even if the other operand
            // is not a constant.
            object value;
            if (lim < 2 && (value = items[1 - lim].AsExpr.ExprValue) != null && !cmpLax(value, value).HasValue)
            {
                node.SetValue(default(BL?));
                return;
            }

            // See if we can reduce to a constant BL value.
            if (lim >= 2)
            {
                if (node.Op != CompareOp.NotEqual)
                {
                    // Note: this loop doesn't work for != when there are more than two operands,
                    // so != is handled separately below.
                    bool isStrict = false;
                    arg = items[0].AsExpr;
                    _host.Assert(arg.ExprType == kind);
                    var valuePrev = arg.ExprValue;
                    _host.Assert(valuePrev != null);
                    for (int i = 1; i < lim; i++)
                    {
                        TokKind tid = node.Operands.Delimiters[i - 1].Kind;
                        _host.Assert(tid == tidLax || tid == tidStrict);

                        if (tid == tidStrict)
                            isStrict = true;

                        arg = items[i].AsExpr;
                        _host.Assert(arg.ExprType == kind);

                        value = arg.ExprValue;
                        _host.Assert(value != null);
                        BL? res = isStrict ? cmpStrict(valuePrev, value) : cmpLax(valuePrev, value);
                        if (res == null || !res.Value)
                        {
                            node.SetValue(false);
                            return;
                        }
                        valuePrev = value;
                        isStrict = false;
                    }
                }
                else
                {
                    // NotEqual is special - it means that the values are all distinct, so comparing adjacent
                    // items is not enough.
                    for (int i = 1; i < lim; i++)
                    {
                        arg = items[i].AsExpr;
                        _host.Assert(arg.ExprType == kind);

                        value = arg.ExprValue;
                        _host.Assert(value != null);
                        for (int j = 0; j < i; j++)
                        {
                            var arg2 = items[j].AsExpr;
                            _host.Assert(arg2.ExprType == kind);

                            var value2 = arg2.ExprValue;
                            _host.Assert(value2 != null);
                            BL? res = cmpStrict(value2, value);
                            if (res == null || !res.Value)
                            {
                                node.SetValue(res);
                                return;
                            }
                        }
                    }
                }

                if (lim == count)
                    node.SetValue(true);
            }
        }

        private sealed class Candidate
        {
            public readonly IFunctionProvider Provider;
            public readonly MethodInfo Method;
            public readonly ExprTypeKind[] Kinds;
            public readonly ExprTypeKind ReturnKind;
            public readonly bool IsVariable;

            public bool MatchesArity(int arity)
            {
                if (!IsVariable)
                    return arity == Kinds.Length;
                Contracts.Assert(Kinds.Length > 0);
                return arity >= Kinds.Length - 1;
            }

            public int Arity
            {
                get { return Kinds.Length; }
            }

            public bool IsIdentity
            {
                get { return Method.ReturnType == typeof(void); }
            }

            public static bool TryGetCandidate(CallNode node, IFunctionProvider provider, MethodInfo meth, Action<string> printError, out Candidate cand)
            {
                cand = default(Candidate);
                if (meth == null)
                    return false;

                // An "identity" function has one parameter and returns void.
                var ps = meth.GetParameters();
                bool isIdent = ps.Length == 1 && meth.ReturnType == typeof(void);

                if (!meth.IsStatic || !meth.IsPublic && !isIdent)
                {
                    // This is an error in the extension functions, not in the user code.
                    printError(string.Format(
                        "Error in ExprTransform: Function '{0}' in namespace '{1}' must be static and public",
                        node.Head.Value, provider.NameSpace));
                    return false;
                }

                // Verify the parameter types.
                bool isVar = false;
                var kinds = new ExprTypeKind[ps.Length];
                for (int i = 0; i < ps.Length; i++)
                {
                    var type = ps[i].ParameterType;
                    if (i == ps.Length - 1 && !isIdent && type.IsArray)
                    {
                        // Last parameter is variable.
                        isVar = true;
                        type = type.GetElementType();
                    }
                    var extCur = ExprNode.ToExprTypeKind(type);
                    if (extCur <= ExprTypeKind.Error || extCur >= ExprTypeKind._Lim)
                    {
                        printError(string.Format(
                            "Error in ExprTransform: Function '{0}' in namespace '{1}' has invalid parameter type '{2}'",
                            node.Head.Value, provider.NameSpace, type));
                        return false;
                    }
                    kinds[i] = extCur;
                }

                // Verify the return type.
                ExprTypeKind kindRet;
                if (isIdent)
                {
                    Contracts.Assert(kinds.Length == 1);
                    kindRet = kinds[0];
                }
                else
                {
                    var extRet = ExprNode.ToExprTypeKind(meth.ReturnType);
                    kindRet = extRet;
                    if (kindRet <= ExprTypeKind.Error || kindRet >= ExprTypeKind._Lim)
                    {
                        printError(string.Format(
                            "Error in ExprTransform: Function '{0}' in namespace '{1}' has invalid return type '{2}'",
                            node.Head.Value, provider.NameSpace, meth.ReturnType));
                        return false;
                    }
                }

                cand = new Candidate(provider, meth, kinds, kindRet, isVar);
                return true;
            }

            private Candidate(IFunctionProvider provider, MethodInfo meth, ExprTypeKind[] kinds, ExprTypeKind kindRet, bool isVar)
            {
                Contracts.AssertValue(provider);
                Contracts.AssertValue(meth);
                Contracts.AssertValue(kinds);
                Provider = provider;
                Method = meth;
                Kinds = kinds;
                ReturnKind = kindRet;
                IsVariable = isVar;
            }

            /// <summary>
            /// Returns whether this candidate is applicable to the given argument types.
            /// </summary>
            public bool IsApplicable(ExprTypeKind[] kinds, out int bad)
            {
                Contracts.Assert(kinds.Length == Kinds.Length || IsVariable && kinds.Length >= Kinds.Length - 1);

                bad = 0;
                int head = IsVariable ? Kinds.Length - 1 : Kinds.Length;

                for (int i = 0; i < head; i++)
                {
                    if (!CanConvert(kinds[i], Kinds[i]))
                        bad++;
                }

                if (IsVariable)
                {
                    // Handle the tail.
                    var kind = Kinds[Kinds.Length - 1];
                    for (int i = head; i < kinds.Length; i++)
                    {
                        if (!CanConvert(kinds[i], kind))
                            bad++;
                    }
                }

                return bad == 0;
            }

            /// <summary>
            /// Returns -1 if 'this' is better than 'other', 0 if they are the same, +1 otherwise.
            /// Non-variable is always better than variable. When both are variable, longer prefix is
            /// better than shorter prefix.
            /// </summary>
            public int CompareSignatures(Candidate other)
            {
                Contracts.AssertValue(other);

                if (IsVariable)
                {
                    if (!other.IsVariable)
                        return +1;
                    if (Kinds.Length != other.Kinds.Length)
                        return Kinds.Length > other.Kinds.Length ? -1 : +1;
                }
                else if (other.IsVariable)
                    return -1;

                int cmp = 0;
                for (int k = 0; k < Kinds.Length; k++)
                {
                    var t1 = Kinds[k];
                    var t2 = other.Kinds[k];
                    if (t1 == t2)
                        continue;
                    if (!CanConvert(t1, t2))
                        return +1;
                    cmp = -1;
                }
                return cmp;
            }
        }

        public override void PostVisit(CallNode node)
        {
            _host.AssertValue(node);

            // Get the argument types and number of arguments.
            var kinds = node.Args.Items.Select(item => item.AsExpr.ExprType).ToArray();
            var arity = kinds.Length;

            // Find the candidates.
            bool hasGoodArity = false;
            var candidates = new List<Candidate>();
            foreach (var prov in _providers)
            {
                if (node.NameSpace != null && prov.NameSpace != node.NameSpace.Value)
                    continue;

                var meths = prov.Lookup(node.Head.Value);
                if (Utils.Size(meths) == 0)
                    continue;

                foreach (var meth in meths)
                {
                    Candidate cand;
                    if (!Candidate.TryGetCandidate(node, prov, meth, _printError, out cand))
                        continue;

                    bool good = cand.MatchesArity(arity);
                    if (hasGoodArity)
                    {
                        // We've seen one or more with good arity, so ignore wrong arity.
                        if (!good)
                            continue;
                    }
                    else if (good)
                    {
                        // This is the first one with good arity.
                        candidates.Clear();
                        hasGoodArity = true;
                    }

                    candidates.Add(cand);
                }
            }

            if (candidates.Count == 0)
            {
                // Unknown function.
                PostError(node.Head, "Unknown function");
                node.SetType(ExprTypeKind.Error);
                return;
            }

            if (!hasGoodArity)
            {
                // No overloads have the target arity. Generate an appropriate error.
                // REVIEW: Will this be good enough with variable arity functions?
                var arities = candidates.Select(c => c.Arity).Distinct().OrderBy(x => x).ToArray();
                if (arities.Length == 1)
                {
                    if (arities[0] == 1)
                        PostError(node, "Expected one argument to function '{1}'", arities[0], node.Head.Value);
                    else
                        PostError(node, "Expected {0} arguments to function '{1}'", arities[0], node.Head.Value);
                }
                else if (arities.Length == 2)
                    PostError(node, "Expected {0} or {1} arguments to function '{2}'", arities[0], arities[1], node.Head.Value);
                else
                    PostError(node, "No overload of function '{0}' takes {1} arguments", node.Head.Value, arity);

                // Set the type of the node. If there is only one possible type, use that, otherwise, use Error.
                var kindsRet = candidates.Select(c => c.ReturnKind).Distinct().ToArray();
                if (kindsRet.Length == 1)
                    node.SetType(kindsRet[0]);
                else
                    node.SetType(ExprTypeKind.Error);
                return;
            }

            // Count applicable candidates and move them to the front.
            int count = 0;
            int minBad = int.MaxValue;
            int icandMinBad = -1;
            for (int i = 0; i < candidates.Count; i++)
            {
                var cand = candidates[i];
                int bad;
                if (cand.IsApplicable(kinds, out bad))
                    candidates[count++] = cand;
                else if (bad < minBad)
                {
                    minBad = bad;
                    icandMinBad = i;
                }
            }
            if (0 < count && count < candidates.Count)
                candidates.RemoveRange(count, candidates.Count - count);
            _host.Assert(candidates.Count > 0);
            _host.Assert(count == 0 || count == candidates.Count);

            // When there are multiple, GetBestOverload picks the one to use and emits an
            // error message if there isn't a unique best answer.
            Candidate best;
            if (count > 1)
                best = GetBestOverload(node, candidates);
            else if (count == 1)
                best = candidates[0];
            else
            {
                _host.Assert(0 <= icandMinBad && icandMinBad < candidates.Count);
                best = candidates[icandMinBad];
                PostError(node, "The best overload of '{0}' has some invalid arguments", node.Head.Value);
            }

            // First convert the arguments to the proper types and get any constant values.
            var args = new object[node.Args.Items.Length];
            bool all = true;
            // For variable, limit the index into best.Kinds to ivMax.
            int ivMax = best.Kinds.Length - 1;
            for (int i = 0; i < node.Args.Items.Length; i++)
            {
                args[i] = Convert(node.Args.Items[i].AsExpr, best.Kinds[Math.Min(i, ivMax)]);
                all &= args[i] != null;
            }

            object res;
            if (best.IsIdentity)
            {
                _host.Assert(!best.IsVariable);
                _host.Assert(best.Arity == 1);
                res = args[0];
            }
            else if (!all)
            {
                res = best.Provider.ResolveToConstant(node.Head.Value, best.Method, args);
                if (res != null && res.GetType() != best.Method.ReturnType)
                {
                    _printError(string.Format(
                        "Error in ExprTransform: Function '{0}' in namespace '{1}' produced wrong constant value type '{2}' vs '{3}'",
                        node.Head.Value, best.Provider.NameSpace, res.GetType(), best.Method.ReturnType));
                    res = null;
                }
            }
            else
            {
                if (best.IsVariable)
                {
                    int head = best.Kinds.Length - 1;
                    int tail = args.Length - head;
                    _host.Assert(tail >= 0);
                    var type = best.Method.GetParameters()[ivMax].ParameterType;
                    _host.Assert(type.IsArray);
                    type = type.GetElementType();
                    Array rest = Array.CreateInstance(type, tail);
                    for (int i = 0; i < tail; i++)
                        rest.SetValue(args[head + i], i);
                    Array.Resize(ref args, head + 1);
                    args[head] = rest;
                }

                res = best.Method.Invoke(null, args);
                _host.Assert(res != null);
                _host.Assert(res.GetType() == best.Method.ReturnType);
            }

            node.SetType(best.ReturnKind, res);
            node.SetMethod(best.Method);
        }

        /// <summary>
        /// Returns whether the given source type can be converted to the given destination type,
        /// for the purposes of function invocation. Returns true if src is null and dst is any
        /// valid type.
        /// </summary>
        private static bool CanConvert(ExprTypeKind src, ExprTypeKind dst)
        {
            // src can be Error, but dst should not be.
            Contracts.Assert(ExprTypeKind.Error <= src && src < ExprTypeKind._Lim);
            Contracts.Assert(ExprTypeKind.Error < dst && dst < ExprTypeKind._Lim);

            if (src == ExprTypeKind.Error)
                return true;

            if (src == dst)
                return true;
            if (src == ExprTypeKind.I4)
                return dst == ExprTypeKind.I8 || dst == ExprTypeKind.R4 || dst == ExprTypeKind.R8;
            if (src == ExprTypeKind.I8)
                return dst == ExprTypeKind.R8;
            if (src == ExprTypeKind.R4)
                return dst == ExprTypeKind.R8;
            return false;
        }

        /// <summary>
        /// Convert the given ExprNode to the given type and get its value, when constant.
        /// </summary>
        private object Convert(ExprNode expr, ExprTypeKind kind)
        {
            switch (kind)
            {
                case ExprTypeKind.BL:
                    {
                        BL? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.BL);
                        return val;
                    }
                case ExprTypeKind.I4:
                    {
                        I4? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.I4);
                        return val;
                    }
                case ExprTypeKind.I8:
                    {
                        I8? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.I8);
                        return val;
                    }
                case ExprTypeKind.R4:
                    {
                        R4? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.R4);
                        return val;
                    }
                case ExprTypeKind.R8:
                    {
                        R8? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.R8);
                        return val;
                    }
                case ExprTypeKind.TX:
                    {
                        TX? val;
                        if (!expr.TryGet(out val))
                            BadArg(expr, ExprTypeKind.TX);
                        return val;
                    }
                default:
                    _host.Assert(false, "Unexpected type in Convert");
                    PostError(expr, "Internal error in Convert");
                    return null;
            }
        }

        /// <summary>
        /// Multiple applicable candidates; pick the best one. We use a simplification of
        /// C#'s rules, only considering the types TX, BL, I4, I8, R4, R8. Basically, parameter
        /// type X is better than parameter type Y if X can be converted to Y. The conversions are:
        ///   I4 => I8, I4 => R4, I4 => R8, I8 => R8, R4 => R8.
        /// </summary>
        private Candidate GetBestOverload(CallNode node, List<Candidate> candidates)
        {
            _host.Assert(Utils.Size(candidates) >= 2);

            var dup1 = default(Candidate);
            var dup2 = default(Candidate);
            for (int i = 0; i < candidates.Count; i++)
            {
                var c1 = candidates[i];
                int dup = -1;
                for (int j = 0; ; j++)
                {
                    if (j == i)
                        continue;

                    if (j >= candidates.Count)
                    {
                        if (dup < 0)
                            return c1;
                        if (dup1 == null)
                        {
                            dup1 = c1;
                            dup2 = candidates[dup];
                        }
                        break;
                    }

                    int cmp = c1.CompareSignatures(candidates[j]);

                    // Break if c1 isn't better.
                    if (cmp > 0)
                        break;
                    if (cmp == 0)
                        dup = j;
                }
            }

            _host.Assert((dup1 != null) == (dup2 != null));
            if (dup1 != null)
            {
                if (dup1.Provider.NameSpace.CompareTo(dup2.Provider.NameSpace) > 0)
                    Utils.Swap(ref dup1, ref dup2);
                PostError(node, "Duplicate candidate functions in namespaces '{0}' and '{1}'",
                    dup1.Provider.NameSpace, dup2.Provider.NameSpace);
            }
            else
                PostError(node, "Ambiguous invocation of function '{0}'", node.Head.Value);

            return dup1 ?? candidates[0];
        }

        public override void PostVisit(ListNode node)
        {
            _host.AssertValue(node);
        }

        public override bool PreVisit(WithNode node)
        {
            _host.AssertValue(node);

            // First bind the value expressions.
            node.Local.Accept(this);

            // Push the with.
            int iwith = _rgwith.Count;
            _rgwith.Add(node);

            // Bind the body.
            node.Body.Accept(this);

            // Pop the var context.
            _host.Assert(_rgwith.Count == iwith + 1);
            _host.Assert(_rgwith[iwith] == node);
            _rgwith.RemoveAt(iwith);
            _host.Assert(_rgwith.Count == iwith);

            node.SetValue(node.Body);

            return false;
        }

        public override void PostVisit(WithNode node)
        {
            _host.Assert(false);
        }

        public override void PostVisit(WithLocalNode node)
        {
            _host.AssertValue(node);
        }

        // This aggregates the type of expr into itemKind. It returns false
        // if an error condition is encountered. This takes into account
        // possible conversions.
        private bool ValidateType(ExprNode expr, ref ExprTypeKind itemKind)
        {
            _host.AssertValue(expr);
            _host.Assert(expr.ExprType != 0);

            ExprTypeKind kind = expr.ExprType;
            switch (kind)
            {
                case ExprTypeKind.Error:
                    _host.Assert(HasErrors);
                    return false;
                case ExprTypeKind.None:
                    return false;
            }

            if (kind == itemKind)
                return true;

            switch (itemKind)
            {
                case ExprTypeKind.Error:
                    // This is the first non-error item type we've seen.
                    _host.Assert(HasErrors);
                    itemKind = kind;
                    return true;
                case ExprTypeKind.None:
                    // This is the first non-error item type we've seen.
                    itemKind = kind;
                    return true;
            }

            ExprTypeKind kindNew;
            if (!CanPromote(true, kind, itemKind, out kindNew))
                return false;

            itemKind = kindNew;
            return true;
        }

        internal static bool CanPromote(bool precise, ExprTypeKind k1, ExprTypeKind k2, out ExprTypeKind res)
        {
            if (k1 == k2)
            {
                res = k1;
                if (res != ExprTypeKind.Error && res != ExprTypeKind.None)
                    return true;
                res = ExprTypeKind.Error;
                return false;
            }

            // Encode numeric types in a two-bit value.
            int i1 = MapKindToIndex(k1);
            int i2 = MapKindToIndex(k2);
            if (i1 < 0 || i2 < 0)
            {
                res = ExprTypeKind.Error;
                return false;
            }

            Contracts.Assert(0 <= i1 && i1 < 4);
            Contracts.Assert(0 <= i2 && i2 < 4);
            Contracts.Assert(i1 != i2);

            // Combine the two two-bit values.
            int index = i1 | (i2 << 2);
            Contracts.Assert(0 <= index && index < 16);
            switch (index)
            {
                // Only integer types -> I8
                case 0x1:
                case 0x4:
                    res = ExprTypeKind.I8;
                    return true;

                // R4 and I4 -> R8 for precise and R4 otherwise.
                case 0x2:
                case 0x8:
                    res = precise ? ExprTypeKind.R8 : ExprTypeKind.R4;
                    return true;

                // At least one RX type and at least one 8-byte type -> R8
                case 0x3:
                case 0x6:
                case 0x7:
                case 0x9:
                case 0xB:
                case 0xC:
                case 0xD:
                case 0xE:
                    res = ExprTypeKind.R8;
                    return true;

                default:
                    Contracts.Assert(false);
                    res = ExprTypeKind.Error;
                    return false;
            }
        }

        /// <summary>
        /// Maps numeric type kinds to an index 0,1,2,3. All others map to -1.
        /// </summary>
        private static int MapKindToIndex(ExprTypeKind kind)
        {
            switch (kind)
            {
                case ExprTypeKind.I4:
                    return 0;
                case ExprTypeKind.I8:
                    return 1;
                case ExprTypeKind.R4:
                    return 2;
                case ExprTypeKind.R8:
                    return 3;
            }
            return -1;
        }
    }

    internal sealed partial class LambdaBinder : NodeVisitor
    {
        // This partial contains stuff needed for equality and ordered comparison.
        private static T Cast<T>(object a)
        {
            Contracts.Assert(a is T);
            return (T)a;
        }

        private delegate BL? Cmp(object a, object b);

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnEqual = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            (a,b) => Cast<BL>(a) == Cast<BL>(b),
            (a,b) => Cast<I4>(a) == Cast<I4>(b),
            (a,b) => Cast<I8>(a) == Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x == y) return true; if (!R4.IsNaN(x) && !R4.IsNaN(y)) return false; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x == y) return true; if (!R8.IsNaN(x) && !R8.IsNaN(y)) return false; return null; },
            (a,b) => Cast<TX>(a).Span.SequenceEqual(Cast<TX>(b).Span),
        };

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnNotEqual = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            (a,b) => Cast<BL>(a) != Cast<BL>(b),
            (a,b) => Cast<I4>(a) != Cast<I4>(b),
            (a,b) => Cast<I8>(a) != Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x == y) return false; if (!R4.IsNaN(x) && !R4.IsNaN(y)) return true; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x == y) return false; if (!R8.IsNaN(x) && !R8.IsNaN(y)) return true; return null; },
            (a,b) => !Cast<TX>(a).Span.SequenceEqual(Cast<TX>(b).Span),
        };

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnLess = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            null,
            (a,b) => Cast<I4>(a) < Cast<I4>(b),
            (a,b) => Cast<I8>(a) < Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x < y) return true; if (x >= y) return false; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x < y) return true; if (x >= y) return false; return null; },
            null,
        };

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnLessEqual = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            null,
            (a,b) => Cast<I4>(a) <= Cast<I4>(b),
            (a,b) => Cast<I8>(a) <= Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x <= y) return true; if (x > y) return false; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x <= y) return true; if (x > y) return false; return null; },
            null,
        };

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnGreater = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            null,
            (a,b) => Cast<I4>(a) > Cast<I4>(b),
            (a,b) => Cast<I8>(a) > Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x > y) return true; if (x <= y) return false; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x > y) return true; if (x <= y) return false; return null; },
            null,
        };

        // Indexed by ExprTypeKind.
        private static readonly Cmp[] _fnGreaterEqual = new Cmp[(int)ExprTypeKind._Lim]
        {
            // None, Error
            null, null,

            null,
            (a,b) => Cast<I4>(a) >= Cast<I4>(b),
            (a,b) => Cast<I8>(a) >= Cast<I8>(b),
            (a,b) => { var x = Cast<R4>(a); var y = Cast<R4>(b); if (x >= y) return true; if (x < y) return false; return null; },
            (a,b) => { var x = Cast<R8>(a); var y = Cast<R8>(b); if (x >= y) return true; if (x < y) return false; return null; },
            null,
        };
    }
}
