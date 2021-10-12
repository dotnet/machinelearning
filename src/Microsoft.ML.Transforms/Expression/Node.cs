// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.IO;
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

    // Operator precedence.
    internal enum Precedence : byte
    {
        None,
        Conditional,
        Coalesce,
        Or,
        And,
        Compare,
        Concat,
        Add,
        Mul,
        Error,
        PrefixUnary,
        Power,
        Postfix,
        Primary,
        Atomic,
    }

    // Types of comparison chains.
    internal enum CompareOp
    {
        Equal,
        NotEqual,
        IncrChain,
        DecrChain
    }

    // Binary operators.
    internal enum BinaryOp
    {
        Coalesce,
        Or,
        And,
        Add,
        Sub,
        Mul,
        Div,
        Mod,
        Power,
        Error
    }

    // Unary operators.
    internal enum UnaryOp
    {
        Not,
        Minus,
    }

    internal enum NodeKind
    {
        Lambda,
        Param,

        Conditional,
        BinaryOp,
        UnaryOp,
        Compare,

        Call,
        List,
        With,
        WithLocal,

        Name,
        Ident,
        BoolLit,
        NumLit,
        StrLit,
    }

    // Note: there are some arrays in LambdaBinder that are indexed by these enum values.
    internal enum ExprTypeKind
    {
        // The order of these matters!
        None,
        Error,

        BL,
        I4,
        I8,
        R4,
        R8,
        TX,

#pragma warning disable MSML_GeneralName // Let's just allow this _ special casing for this internal enum.
        _Lim,
#pragma warning restore MSML_GeneralName
        Float = R4
    }

    internal abstract class NodeVisitor
    {
        // Visit methods for leaf node types.
        public abstract void Visit(BoolLitNode node);
        public abstract void Visit(StrLitNode node);
        public abstract void Visit(NumLitNode node);
        public abstract void Visit(NameNode node);
        public abstract void Visit(IdentNode node);
        public abstract void Visit(ParamNode node);

        // Visit methods for non-leaf node types.
        // If PreVisit returns true, the children are visited and PostVisit is called.
        public virtual bool PreVisit(LambdaNode node) { return true; }
        public virtual bool PreVisit(UnaryOpNode node) { return true; }
        public virtual bool PreVisit(BinaryOpNode node) { return true; }
        public virtual bool PreVisit(ConditionalNode node) { return true; }
        public virtual bool PreVisit(CompareNode node) { return true; }
        public virtual bool PreVisit(CallNode node) { return true; }
        public virtual bool PreVisit(ListNode node) { return true; }
        public virtual bool PreVisit(WithNode node) { return true; }
        public virtual bool PreVisit(WithLocalNode node) { return true; }

        public abstract void PostVisit(LambdaNode node);
        public abstract void PostVisit(UnaryOpNode node);
        public abstract void PostVisit(BinaryOpNode node);
        public abstract void PostVisit(ConditionalNode node);
        public abstract void PostVisit(CompareNode node);
        public abstract void PostVisit(CallNode node);
        public abstract void PostVisit(ListNode node);
        public abstract void PostVisit(WithNode node);
        public abstract void PostVisit(WithLocalNode node);
    }

    internal abstract class PreVisitor : NodeVisitor
    {
        // Visit methods for non-leaf node types.
        public abstract void Visit(LambdaNode node);
        public abstract void Visit(UnaryOpNode node);
        public abstract void Visit(BinaryOpNode node);
        public abstract void Visit(ConditionalNode node);
        public abstract void Visit(CompareNode node);
        public abstract void Visit(CallNode node);
        public abstract void Visit(ListNode node);
        public abstract void Visit(WithNode node);
        public abstract void Visit(WithLocalNode node);

        // PreVisit and PostVisit methods for non-leaf node types.
        public override bool PreVisit(LambdaNode node) { Visit(node); return false; }
        public override bool PreVisit(UnaryOpNode node) { Visit(node); return false; }
        public override bool PreVisit(BinaryOpNode node) { Visit(node); return false; }
        public override bool PreVisit(ConditionalNode node) { Visit(node); return false; }
        public override bool PreVisit(CompareNode node) { Visit(node); return false; }
        public override bool PreVisit(CallNode node) { Visit(node); return false; }
        public override bool PreVisit(ListNode node) { Visit(node); return false; }
        public override bool PreVisit(WithNode node) { Visit(node); return false; }
        public override bool PreVisit(WithLocalNode node) { Visit(node); return false; }

        public override void PostVisit(LambdaNode node) { Contracts.Assert(false); }
        public override void PostVisit(UnaryOpNode node) { Contracts.Assert(false); }
        public override void PostVisit(BinaryOpNode node) { Contracts.Assert(false); }
        public override void PostVisit(ConditionalNode node) { Contracts.Assert(false); }
        public override void PostVisit(CompareNode node) { Contracts.Assert(false); }
        public override void PostVisit(CallNode node) { Contracts.Assert(false); }
        public override void PostVisit(ListNode node) { Contracts.Assert(false); }
        public override void PostVisit(WithNode node) { Contracts.Assert(false); }
        public override void PostVisit(WithLocalNode node) { Contracts.Assert(false); }
    }

    internal abstract class ExprVisitor : NodeVisitor
    {
        // This just provides default implementations for non-expr related node types.
        public override void Visit(NameNode node) { Contracts.Assert(false); }
        public override void Visit(ParamNode node) { Contracts.Assert(false); }

        public override void PostVisit(LambdaNode node) { Contracts.Assert(false); }
    }

    // Base class for all parse nodes.
    internal abstract class Node
    {
        public readonly Token Token;

        protected Node(Token tok)
        {
            Contracts.AssertValue(tok);
            Token = tok;
        }

        public abstract NodeKind Kind { get; }
        public abstract void Accept(NodeVisitor visitor);

        #region AsXxx and TestXxx members

        private T Cast<T>() where T : Node
        {
            Contracts.Assert(false);
            return (T)this;
        }

        // TestXxx returns null if "this" is not of the correct type.
        // In contrast AsXxx asserts that the default implementation is not being
        // used (the derived type should override). TestXxx is for when you don't know
        // and you will check the return result for null. AXxx is for when you do know.
        public virtual LambdaNode AsPredicate { get { return Cast<LambdaNode>(); } }
        public virtual LambdaNode TestPredicate { get { return null; } }
        public virtual ParamNode AsParam { get { return Cast<ParamNode>(); } }
        public virtual ParamNode TestParam { get { return null; } }
        public virtual ConditionalNode AsConditional { get { return Cast<ConditionalNode>(); } }
        public virtual ConditionalNode TestConditional { get { return null; } }
        public virtual BinaryOpNode AsBinaryOp { get { return Cast<BinaryOpNode>(); } }
        public virtual BinaryOpNode TestBinaryOp { get { return null; } }
        public virtual UnaryOpNode AsUnaryOp { get { return Cast<UnaryOpNode>(); } }
        public virtual UnaryOpNode TestUnaryOp { get { return null; } }
        public virtual CompareNode AsCompare { get { return Cast<CompareNode>(); } }
        public virtual CompareNode TestCompare { get { return null; } }
        public virtual CallNode AsCall { get { return Cast<CallNode>(); } }
        public virtual CallNode TestCall { get { return null; } }
        public virtual ListNode AsList { get { return Cast<ListNode>(); } }
        public virtual ListNode TestList { get { return null; } }
        public virtual WithNode AsWith { get { return Cast<WithNode>(); } }
        public virtual WithNode TestWith { get { return null; } }
        public virtual WithLocalNode AsWithLocal { get { return Cast<WithLocalNode>(); } }
        public virtual WithLocalNode TestWithLocal { get { return null; } }
        public virtual NameNode AsName { get { return Cast<NameNode>(); } }
        public virtual NameNode TestName { get { return null; } }
        public virtual IdentNode AsIdent { get { return Cast<IdentNode>(); } }
        public virtual IdentNode TestIdent { get { return null; } }
        public virtual BoolLitNode AsBoolLit { get { return Cast<BoolLitNode>(); } }
        public virtual BoolLitNode TestBoolLit { get { return null; } }
        public virtual NumLitNode AsNumLit { get { return Cast<NumLitNode>(); } }
        public virtual NumLitNode TestNumLit { get { return null; } }
        public virtual StrLitNode AsStrLit { get { return Cast<StrLitNode>(); } }
        public virtual StrLitNode TestStrLit { get { return null; } }

        // Non-leaf types.
        public virtual ExprNode AsExpr { get { return Cast<ExprNode>(); } }
        public virtual ExprNode TestExpr { get { return null; } }

        #endregion CastXxx and TestXxx members

        public override string ToString()
        {
            using (var wrt = new StringWriter())
            {
                NodePrinter.Print(this, wrt);
                return wrt.ToString();
            }
        }
    }

    internal abstract class ExprNode : Node
    {
        protected ExprNode(Token tok)
            : base(tok)
        {
        }

        public override ExprNode AsExpr { get { return this; } }
        public override ExprNode TestExpr { get { return this; } }

        public ExprTypeKind ExprType { get; private set; }
        public object ExprValue { get; private set; }

        private bool IsSimple(ExprTypeKind kind)
        {
            Contracts.Assert(ExprType != 0);
            return ExprType == kind;
        }

        public bool HasType { get { return ExprTypeKind.Error < ExprType && ExprType < ExprTypeKind._Lim; } }
        public bool IsNone { get { return ExprType == ExprTypeKind.None; } }
        public bool IsError { get { return ExprType == ExprTypeKind.Error; } }

        public bool IsBool { get { return IsSimple(ExprTypeKind.BL); } }
        public bool IsNumber
        {
            get
            {
                return
                    IsSimple(ExprTypeKind.I4) || IsSimple(ExprTypeKind.I8) ||
                    IsSimple(ExprTypeKind.R4) || IsSimple(ExprTypeKind.R8);
            }
        }
        public bool IsI4 { get { return IsSimple(ExprTypeKind.I4); } }
        public bool IsI8 { get { return IsSimple(ExprTypeKind.I8); } }
        public bool IsRx { get { return IsSimple(ExprTypeKind.R4) || IsSimple(ExprTypeKind.R8); } }
        public bool IsR4 { get { return IsSimple(ExprTypeKind.R4); } }
        public bool IsR8 { get { return IsSimple(ExprTypeKind.R8); } }
        public bool IsTX { get { return IsSimple(ExprTypeKind.TX); } }

        public ExprTypeKind SrcKind { get; private set; }
        public bool NeedsConversion
        {
            get
            {
#if DEBUG
                // Assert that the conversion is valid.
                if (SrcKind != ExprType)
                {
                    ExprTypeKind kind;
                    bool tmp = LambdaBinder.CanPromote(false, SrcKind, ExprType, out kind);
                    Contracts.Assert(tmp && kind == ExprType);
                }
#endif
                return SrcKind != ExprType;
            }
        }

        public void SetType(ExprTypeKind kind)
        {
            Contracts.Assert(kind != 0);
            Contracts.Assert(ExprValue == null);
            Contracts.Assert(ExprType == 0 || ExprType == kind);
            Contracts.Assert(SrcKind == ExprType);
            ExprType = kind;
            SrcKind = kind;
        }

        public void SetType(ExprTypeKind kind, object value)
        {
            Contracts.Assert(kind != 0);
            Contracts.Assert(value == null || value.GetType() == ToSysType(kind));
            Contracts.Assert(ExprValue == null);
            Contracts.Assert(ExprType == 0 || ExprType == kind);
            Contracts.Assert(SrcKind == ExprType);
            ExprType = kind;
            SrcKind = kind;
            ExprValue = value;
        }

        internal static Type ToSysType(ExprTypeKind kind)
        {
            switch (kind)
            {
                case ExprTypeKind.BL:
                    return typeof(BL);
                case ExprTypeKind.I4:
                    return typeof(I4);
                case ExprTypeKind.I8:
                    return typeof(I8);
                case ExprTypeKind.R4:
                    return typeof(R4);
                case ExprTypeKind.R8:
                    return typeof(R8);
                case ExprTypeKind.TX:
                    return typeof(TX);
                default:
                    return null;
            }
        }

        internal static ExprTypeKind ToExprTypeKind(Type type)
        {
            if (type == typeof(BL))
                return ExprTypeKind.BL;
            if (type == typeof(I4))
                return ExprTypeKind.I4;
            if (type == typeof(I8))
                return ExprTypeKind.I8;
            if (type == typeof(R4))
                return ExprTypeKind.R4;
            if (type == typeof(R8))
                return ExprTypeKind.R8;
            if (type == typeof(TX))
                return ExprTypeKind.TX;
            return ExprTypeKind.Error;
        }

        public void SetValue(ExprNode expr)
        {
            Contracts.AssertValue(expr);
            Contracts.Assert(expr.ExprType != 0);
            SetType(expr.ExprType);
            ExprValue = expr.ExprValue;
        }

        public void SetValue(BL value)
        {
            SetType(ExprTypeKind.BL);
            ExprValue = value;
        }

        public void SetValue(BL? value)
        {
            SetType(ExprTypeKind.BL);
            ExprValue = value;
        }

        public void SetValue(I4 value)
        {
            SetType(ExprTypeKind.I4);
            ExprValue = value;
        }

        public void SetValue(I4? value)
        {
            SetType(ExprTypeKind.I4);
            ExprValue = value;
        }

        public void SetValue(I8 value)
        {
            SetType(ExprTypeKind.I8);
            ExprValue = value;
        }

        public void SetValue(I8? value)
        {
            SetType(ExprTypeKind.I8);
            ExprValue = value;
        }

        public void SetValue(R4 value)
        {
            SetType(ExprTypeKind.R4);
            ExprValue = value;
        }

        public void SetValue(R4? value)
        {
            SetType(ExprTypeKind.R4);
            ExprValue = value;
        }

        public void SetValue(R8 value)
        {
            SetType(ExprTypeKind.R8);
            ExprValue = value;
        }

        public void SetValue(R8? value)
        {
            SetType(ExprTypeKind.R8);
            ExprValue = value;
        }

        public void SetValue(TX value)
        {
            SetType(ExprTypeKind.TX);
            ExprValue = value;
        }

        public void SetValue(TX? value)
        {
            SetType(ExprTypeKind.TX);
            ExprValue = value;
        }

        public void Convert(ExprTypeKind kind)
        {
            Contracts.Assert(HasType);

            if (kind == ExprType)
                return;

            Contracts.Assert(SrcKind == ExprType);
            switch (kind)
            {
                case ExprTypeKind.I8:
                    Contracts.Assert(ExprType == ExprTypeKind.I4);
                    if (ExprValue != null)
                    {
                        Contracts.Assert(ExprValue is I4);
                        ExprValue = (I8)(I4)ExprValue;
                    }
                    break;
                case ExprTypeKind.R4:
                    Contracts.Assert(ExprType == ExprTypeKind.I4);
                    if (ExprValue != null)
                    {
                        Contracts.Assert(ExprValue is I4);
                        ExprValue = (R4)(I4)ExprValue;
                    }
                    break;
                case ExprTypeKind.R8:
                    Contracts.Assert(ExprType == ExprTypeKind.I4 || ExprType == ExprTypeKind.I8 ||
                        ExprType == ExprTypeKind.R4);
                    if (ExprValue != null)
                    {
                        if (ExprType == ExprTypeKind.I4)
                        {
                            Contracts.Assert(ExprValue is I4);
                            ExprValue = (R8)(I4)ExprValue;
                        }
                        else if (ExprType == ExprTypeKind.I8)
                        {
                            Contracts.Assert(ExprValue is I8);
                            ExprValue = (R8)(I8)ExprValue;
                        }
                        else
                        {
                            Contracts.Assert(ExprValue is R4);
                            ExprValue = (R8)(R4)ExprValue;
                        }
                    }
                    break;
            }

            // Set the new type.
            ExprType = kind;
        }

        public bool TryGet(out BL? value)
        {
            if (IsBool)
            {
                value = (BL?)ExprValue;
                return true;
            }
            value = null;
            return false;
        }

        public bool TryGet(out I4? value)
        {
            if (IsI4)
            {
                value = (I4?)ExprValue;
                return true;
            }
            value = null;
            return false;
        }

        public bool TryGet(out I8? value)
        {
            switch (ExprType)
            {
                default:
                    value = null;
                    return false;
                case ExprTypeKind.I4:
                case ExprTypeKind.I8:
                    break;
            }
            Convert(ExprTypeKind.I8);
            value = (I8?)ExprValue;
            return true;
        }

        public bool TryGet(out R4? value)
        {
            switch (ExprType)
            {
                default:
                    value = null;
                    return false;
                case ExprTypeKind.I4:
                case ExprTypeKind.R4:
                    break;
            }
            Convert(ExprTypeKind.R4);
            value = (R4?)ExprValue;
            return true;
        }

        public bool TryGet(out R8? value)
        {
            switch (ExprType)
            {
                default:
                    value = null;
                    return false;
                case ExprTypeKind.I4:
                case ExprTypeKind.I8:
                case ExprTypeKind.R4:
                case ExprTypeKind.R8:
                    break;
            }
            Convert(ExprTypeKind.R8);
            value = (R8?)ExprValue;
            return true;
        }

        public bool TryGet(out TX? value)
        {
            if (IsTX)
            {
                value = (TX?)ExprValue;
                return true;
            }
            value = null;
            return false;
        }
    }

    internal sealed class LambdaNode : Node
    {
        public readonly ParamNode[] Vars;
        public readonly ExprNode Expr;

        public DataViewType ResultType;

        public LambdaNode(Token tok, ParamNode[] vars, ExprNode expr)
            : base(tok)
        {
            Contracts.AssertNonEmpty(vars);
            Contracts.AssertValue(expr);
            Vars = vars;
            Expr = expr;
        }

        public override NodeKind Kind { get { return NodeKind.Lambda; } }
        public override LambdaNode AsPredicate { get { return this; } }
        public override LambdaNode TestPredicate { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                foreach (var v in Vars)
                    v.Accept(visitor);
                Expr.Accept(visitor);
                visitor.PostVisit(this);
            }
        }

        public ParamNode FindParam(string name)
        {
            foreach (var v in Vars)
            {
                if (v.Name == name)
                    return v;
            }
            return null;
        }
    }

    internal sealed class ParamNode : Node
    {
        public readonly string Name;
        public readonly int Index;
        public readonly DataViewType Type;
        public ExprTypeKind ExprType;

        public ParamNode(Token tok, string name, int index, DataViewType type)
            : base(tok)
        {
            Contracts.AssertNonEmpty(name);
            Contracts.Assert(index >= 0);
            Contracts.AssertValueOrNull(type);
            Name = name;
            Index = index;
            Type = type;

            if (type == null)
                ExprType = ExprTypeKind.Error;
            else if (type is TextDataViewType)
                ExprType = ExprTypeKind.TX;
            else if (type is BooleanDataViewType)
                ExprType = ExprTypeKind.BL;
            else if (type == NumberDataViewType.Int32)
                ExprType = ExprTypeKind.I4;
            else if (type == NumberDataViewType.Int64)
                ExprType = ExprTypeKind.I8;
            else if (type == NumberDataViewType.Single)
                ExprType = ExprTypeKind.R4;
            else if (type == NumberDataViewType.Double)
                ExprType = ExprTypeKind.R8;
        }

        public override NodeKind Kind { get { return NodeKind.Param; } }
        public override ParamNode AsParam { get { return this; } }
        public override ParamNode TestParam { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    // A NameNode identifies the name of something. An IdentNode is an expression node
    // consisting of an identifier.
    internal sealed class NameNode : Node
    {
        public readonly string Value;

        public NameNode(IdentToken tok)
            : base(tok)
        {
            Contracts.AssertNonEmpty(tok.Value);
            Value = tok.Value;
        }

        public override NodeKind Kind { get { return NodeKind.Name; } }
        public override NameNode AsName { get { return this; } }
        public override NameNode TestName { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    internal sealed class IdentNode : ExprNode
    {
        public readonly string Value;
        public readonly bool IsMissing;

        // If this ident node is a reference to another node, this
        // is set to the node (by the Binder).
        public Node Referent;

        public IdentNode(IdentToken tok)
            : base(tok)
        {
            Contracts.AssertNonEmpty(tok.Value);
            Value = tok.Value;
        }

        public IdentNode(Token tok, string value, bool missing = false)
            : base(tok)
        {
            Contracts.AssertNonEmpty(value);
            Value = value;
            IsMissing = missing;
        }

        public override NodeKind Kind { get { return NodeKind.Ident; } }
        public override IdentNode AsIdent { get { return this; } }
        public override IdentNode TestIdent { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    internal sealed class NumLitNode : ExprNode
    {
        public NumLitNode(NumLitToken tok)
            : base(tok)
        {
            switch (tok.Kind)
            {
                default:
                    Contracts.Assert(false);
                    SetType(ExprTypeKind.Error);
                    return;

                case TokKind.FltLit:
                    SetValue(tok.As<FltLitToken>().Value);
                    return;

                case TokKind.DblLit:
                    {
                        var t = tok.As<DblLitToken>();
                        if (t.HasSuffix)
                            SetValue(t.Value);
                        else
                            SetValue((float)t.Value);
                    }
                    return;

                case TokKind.IntLit:
                    break;
            }
            Contracts.Assert(tok.Kind == TokKind.IntLit);

            var ilt = tok.As<IntLitToken>();
            var uu = ilt.Value;
            bool lng = (ilt.IntKind & IntLitKind.Lng) != 0;
            bool uns = (ilt.IntKind & IntLitKind.Uns) != 0;

            // If it is in I4 range or it is hex and in uint range, use I4.
            // Otherwise, if it is in I8 range or it is hex, use I8. Otherwise, error.
            // REVIEW: Should we do NA instead of error?
            if (!lng && (uu <= I4.MaxValue || uu <= uint.MaxValue && ilt.IsHex && !uns))
                SetValue((I4)uu);
            else if (uu <= I8.MaxValue || ilt.IsHex && !uns)
                SetValue((I8)uu);
            else
                SetType(ExprTypeKind.Error);
        }

        public NumLitToken Value
        {
            get { return Token.As<NumLitToken>(); }
        }

        public override NodeKind Kind { get { return NodeKind.NumLit; } }
        public override NumLitNode AsNumLit { get { return this; } }
        public override NumLitNode TestNumLit { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    internal sealed class StrLitNode : ExprNode
    {
        public readonly TX Value;

        public StrLitNode(StrLitToken tok)
            : base(tok)
        {
            Contracts.AssertValue(tok.Value);
            Value = tok.Value.AsMemory();
            SetValue(Value);
        }

        public override NodeKind Kind { get { return NodeKind.StrLit; } }
        public override StrLitNode AsStrLit { get { return this; } }
        public override StrLitNode TestStrLit { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    internal sealed class BoolLitNode : ExprNode
    {
        public BoolLitNode(Token tok)
            : base(tok)
        {
            Contracts.AssertValue(tok);
            Contracts.Assert(tok.Kind == TokKind.True || tok.Kind == TokKind.False);
            SetValue(tok.Kind == TokKind.True ? true : false);
        }

        public bool Value { get { return Token.Kind == TokKind.True; } }

        public override NodeKind Kind { get { return NodeKind.BoolLit; } }
        public override BoolLitNode AsBoolLit { get { return this; } }
        public override BoolLitNode TestBoolLit { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            visitor.Visit(this);
        }
    }

    internal sealed class UnaryOpNode : ExprNode
    {
        public readonly ExprNode Arg;
        public readonly UnaryOp Op;

        public UnaryOpNode(Token tok, UnaryOp op, ExprNode arg)
            : base(tok)
        {
            Contracts.AssertValue(arg);
            Arg = arg;
            Op = op;
        }

        public override NodeKind Kind { get { return NodeKind.UnaryOp; } }
        public override UnaryOpNode AsUnaryOp { get { return this; } }
        public override UnaryOpNode TestUnaryOp { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                Arg.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }

    internal sealed class BinaryOpNode : ExprNode
    {
        public readonly ExprNode Left;
        public readonly ExprNode Right;
        public readonly BinaryOp Op;

        public bool ReduceToLeft;
        public bool ReduceToRight;

        public BinaryOpNode(Token tok, BinaryOp op, ExprNode left, ExprNode right)
            : base(tok)
        {
            Contracts.AssertValue(left);
            Contracts.AssertValue(right);
            Left = left;
            Right = right;
            Op = op;
        }

        public override NodeKind Kind { get { return NodeKind.BinaryOp; } }
        public override BinaryOpNode AsBinaryOp { get { return this; } }
        public override BinaryOpNode TestBinaryOp { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                Left.Accept(visitor);
                Right.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }

    /// <summary>
    /// Node for the ternary conditional operator.
    /// </summary>
    internal sealed class ConditionalNode : ExprNode
    {
        public readonly ExprNode Cond;
        public readonly ExprNode Left;
        public readonly Token TokColon;
        public readonly ExprNode Right;

        public ConditionalNode(Token tok, ExprNode cond, ExprNode left, Token tokColon, ExprNode right)
            : base(tok)
        {
            Contracts.AssertValue(cond);
            Contracts.AssertValue(left);
            Contracts.AssertValueOrNull(tokColon);
            Contracts.AssertValue(right);
            Cond = cond;
            Left = left;
            Right = right;
            TokColon = tokColon;
        }

        public override NodeKind Kind { get { return NodeKind.Conditional; } }
        public override ConditionalNode AsConditional { get { return this; } }
        public override ConditionalNode TestConditional { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                Cond.Accept(visitor);
                Left.Accept(visitor);
                Right.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }

    internal sealed class ListNode : Node
    {
        public readonly Node[] Items;
        public readonly Token[] Delimiters;

        // Assumes ownership of items array and the delimiters array.
        public ListNode(Token tok, Node[] items, Token[] delimiters)
            : base(tok)
        {
            Contracts.AssertValue(items);
            Contracts.AssertValueOrNull(delimiters);
            Contracts.Assert(delimiters == null || delimiters.Length == items.Length - 1);
            Items = items;
            Delimiters = delimiters;
        }

        public override NodeKind Kind { get { return NodeKind.List; } }
        public override ListNode AsList { get { return this; } }
        public override ListNode TestList { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                foreach (var item in Items)
                {
                    Contracts.AssertValue(item);
                    item.Accept(visitor);
                }
                visitor.PostVisit(this);
            }
        }
    }

    internal sealed class CallNode : ExprNode
    {
        // NameSpace and Dot can be null.
        public readonly NameNode NameSpace;
        public readonly Token Dot;
        // Head and Args will never be null.
        public readonly NameNode Head;
        public readonly ListNode Args;
        // CloseToken can be null.
        public readonly Token CloseToken;

        public MethodInfo Method { get; private set; }

        public CallNode(Token tok, NameNode head, ListNode args, Token tokClose)
            : base(tok)
        {
            Contracts.AssertValue(head);
            Contracts.AssertValue(args);
            Contracts.AssertValueOrNull(tokClose);
            Head = head;
            Args = args;
            CloseToken = tokClose;
        }

        public CallNode(Token tok, NameNode ns, Token dot, NameNode head, ListNode args, Token tokClose)
            : base(tok)
        {
            Contracts.AssertValue(ns);
            Contracts.AssertValue(dot);
            Contracts.AssertValue(head);
            Contracts.AssertValue(args);
            Contracts.AssertValueOrNull(tokClose);
            NameSpace = ns;
            Dot = dot;
            Head = head;
            Args = args;
            CloseToken = tokClose;
        }

        public override NodeKind Kind { get { return NodeKind.Call; } }
        public override CallNode AsCall { get { return this; } }
        public override CallNode TestCall { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                if (NameSpace != null)
                    NameSpace.Accept(visitor);
                Head.Accept(visitor);
                Args.Accept(visitor);
                visitor.PostVisit(this);
            }
        }

        public void SetMethod(MethodInfo meth)
        {
#if DEBUG
            var argCount = Args.Items.Length;
            var ps = meth.GetParameters();
            if (Utils.Size(ps) > 0 && ps[ps.Length - 1].ParameterType.IsArray)
            {
                // Variable case.
                Contracts.Assert(argCount >= ps.Length - 1);
                Contracts.Assert(meth.ReturnType != typeof(void));
            }
            else
                Contracts.Assert(Utils.Size(ps) == argCount);
#endif
            Method = meth;
        }
    }

    internal sealed class CompareNode : ExprNode
    {
        public readonly CompareOp Op;
        public readonly ListNode Operands;
        public readonly TokKind TidStrict;
        public readonly TokKind TidLax;

        public ExprTypeKind ArgTypeKind;

        public CompareNode(Token tok, CompareOp op, ListNode operands)
            : base(tok)
        {
            Contracts.AssertValue(operands);
            Contracts.Assert(operands.Items.Length >= 2);
            Contracts.AssertValue(operands.Delimiters);
            Contracts.Assert(operands.Delimiters.Length == operands.Items.Length - 1);
            Op = op;
            Operands = operands;

            switch (op)
            {
                default:
                    Contracts.Assert(false);
                    goto case CompareOp.Equal;
                case CompareOp.Equal:
                    TidLax = TokKind.Equ;
                    TidStrict = TokKind.EquEqu;
                    break;
                case CompareOp.NotEqual:
                    TidLax = TokKind.LssGrt;
                    TidStrict = TokKind.BngEqu;
                    break;
                case CompareOp.IncrChain:
                    TidLax = TokKind.LssEqu;
                    TidStrict = TokKind.Lss;
                    break;
                case CompareOp.DecrChain:
                    TidLax = TokKind.GrtEqu;
                    TidStrict = TokKind.Grt;
                    break;
            }
        }

        public override NodeKind Kind { get { return NodeKind.Compare; } }
        public override CompareNode AsCompare { get { return this; } }
        public override CompareNode TestCompare { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            Contracts.AssertValue(visitor);
            if (visitor.PreVisit(this))
            {
                Operands.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }

    /// <summary>
    /// The parse node for a "with" expression. The grammar is:
    ///
    ///   WithNode :
    ///     with ( WithLocalNode [ , WithLocalNode ]* ; Expr )
    ///
    /// Note that a with expression with multiple WithLocalNodes gets expanded to multiple
    /// nested WithNodes. This makes the code much simpler and easily allows a with-local to
    /// reference all previous with-locals.
    /// </summary>
    internal sealed class WithNode : ExprNode
    {
        public readonly WithLocalNode Local;
        public readonly ExprNode Body;

        public WithNode(Token tok, WithLocalNode local, ExprNode body)
            : base(tok)
        {
            Contracts.AssertValue(local);
            Contracts.AssertValue(body);
            Local = local;
            Body = body;
        }

        public override NodeKind Kind { get { return NodeKind.With; } }
        public override WithNode AsWith { get { return this; } }
        public override WithNode TestWith { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            if (visitor.PreVisit(this))
            {
                Local.Accept(visitor);
                Body.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }

    /// <summary>
    /// The with-expression local assignment node. This contains both the name of the local and the
    /// value expression.
    /// </summary>
    internal sealed class WithLocalNode : Node
    {
        public readonly string Name;
        public readonly ExprNode Value;

        // Records whether this variable is used by the code. Set by the binder.
        public int UseCount;

        // Index is assigned and used by the code generator. It indicates which local this
        // is stored in, when UseCount > 1. Otherwise, it is -1. Note that when UseCount == 1,
        // we inline code-gen for this local.
        public int Index;
        // The number of times the code for this local was generated.
        public int GenCount;

        public WithLocalNode(Token tok, string name, ExprNode value)
            : base(tok)
        {
            Contracts.AssertValue(name);
            Contracts.AssertValue(value);
            Name = name;
            Value = value;
            Index = -1;
        }

        public override NodeKind Kind { get { return NodeKind.WithLocal; } }
        public override WithLocalNode AsWithLocal { get { return this; } }
        public override WithLocalNode TestWithLocal { get { return this; } }

        public override void Accept(NodeVisitor visitor)
        {
            if (visitor.PreVisit(this))
            {
                Value.Accept(visitor);
                visitor.PostVisit(this);
            }
        }
    }
}
