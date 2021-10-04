// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.CodeDom.Compiler;
using System.IO;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Transforms
{
    using BL = System.Boolean;
    using I4 = System.Int32;
    using I8 = System.Int64;
    using R4 = Single;
    using R8 = Double;
    using TX = ReadOnlyMemory<char>;

    // Simple pretty-printing visitor
    internal sealed class NodePrinter : PreVisitor
    {
        private readonly bool _showTypes;
        private readonly bool _showValues;
        private readonly IndentedTextWriter _wrt;

        private NodePrinter(IndentedTextWriter wrt, bool showTypes, bool showValues)
        {
            Contracts.AssertValue(wrt);

            _showTypes = showTypes;
            _showValues = showValues;
            _wrt = wrt;
        }

        // Public entry point for prettyprinting TEXL parse trees
        public static void Print(Node node, TextWriter writer, bool showTypes = false, bool showValues = false)
        {
            Contracts.AssertValue(node);

            var wrt = new IndentedTextWriter(writer, "  ");
            NodePrinter printer = new NodePrinter(wrt, showTypes, showValues);
            node.Accept(printer);
        }

        private bool NeedParensLeft(Precedence precLeft, Precedence precOp)
        {
            if (precLeft < precOp)
                return true;
            if (precLeft > precOp)
                return false;

            // Power is right associative.
            return precOp == Precedence.Power;
        }

        private bool NeedParensRight(Precedence precOp, Precedence precRight)
        {
            if (precOp == Precedence.Postfix)
            {
                // Indexing is the only postfix operator, and it never
                // needs parens around the right operand.
                return false;
            }

            if (precOp > precRight)
                return true;
            if (precOp < precRight)
                return false;

            // Power is right associative.
            return precOp != Precedence.Power;
        }

        private Precedence GetPrec(Node node)
        {
            Contracts.Assert(node is ExprNode);

            switch (node.Kind)
            {
                case NodeKind.BinaryOp:
                    return GetPrec(node.AsBinaryOp.Op);

                case NodeKind.UnaryOp:
                    return Precedence.PrefixUnary;

                case NodeKind.Compare:
                    return Precedence.Compare;

                case NodeKind.Call:
                case NodeKind.With:
                    return Precedence.Primary;

                case NodeKind.Ident:
                case NodeKind.BoolLit:
                case NodeKind.NumLit:
                case NodeKind.StrLit:
                    return Precedence.Atomic;

                default:
                    Contracts.Assert(false, "Unexpected node kind in GetPrec - should only see ExprNode kinds");
                    return Precedence.None;
            }
        }

        private Precedence GetPrec(BinaryOp op)
        {
            switch (op)
            {
                case BinaryOp.Or:
                    return Precedence.Or;
                case BinaryOp.And:
                    return Precedence.And;
                case BinaryOp.Add:
                case BinaryOp.Sub:
                    return Precedence.Add;
                case BinaryOp.Mul:
                case BinaryOp.Div:
                case BinaryOp.Mod:
                    return Precedence.Mul;
                case BinaryOp.Power:
                    return Precedence.Power;
                case BinaryOp.Error:
                    return Precedence.None;
                default:
                    Contracts.Assert(false);
                    return Precedence.None;
            }
        }

        private string GetString(BinaryOp op)
        {
            switch (op)
            {
                case BinaryOp.Or:
                    return " or ";
                case BinaryOp.And:
                    return " and ";
                case BinaryOp.Add:
                    return " + ";
                case BinaryOp.Sub:
                    return " - ";
                case BinaryOp.Mul:
                    return " * ";
                case BinaryOp.Div:
                    return " / ";
                case BinaryOp.Mod:
                    return " % ";
                case BinaryOp.Power:
                    return " ^ ";
                case BinaryOp.Error:
                    return " <err> ";
                default:
                    Contracts.Assert(false);
                    return " <bad> ";
            }
        }

        private string GetString(UnaryOp op)
        {
            switch (op)
            {
                case UnaryOp.Not:
                    return "not ";
                case UnaryOp.Minus:
                    return "-";
                default:
                    Contracts.Assert(false);
                    return "<bad> ";
            }
        }

        private string GetString(TokKind tidCompare)
        {
            switch (tidCompare)
            {
                case TokKind.Equ:
                    return " = ";
                case TokKind.EquEqu:
                    return " == ";
                case TokKind.LssGrt:
                    return " <> ";
                case TokKind.BngEqu:
                    return " != ";
                case TokKind.Lss:
                    return " < ";
                case TokKind.LssEqu:
                    return " <= ";
                case TokKind.GrtEqu:
                    return " >= ";
                case TokKind.Grt:
                    return " > ";

                default:
                    Contracts.Assert(false);
                    return " <bad> ";
            }
        }

        private bool TryShowValue(ExprNode node)
        {
            if (!_showValues)
                return false;
            if (node.ExprValue == null)
                return false;

            ShowValueCore(node);
            ShowType(node);

            return true;
        }

        private void ShowValueCore(ExprNode node)
        {
            Contracts.AssertValue(node);
            Contracts.AssertValue(node.ExprValue);

            var value = node.ExprValue;
            switch (node.ExprType)
            {
                case ExprTypeKind.I4:
                    Show((I4)value);
                    break;
                case ExprTypeKind.I8:
                    Show((I8)value);
                    break;
                case ExprTypeKind.R4:
                    Show((R4)value);
                    break;
                case ExprTypeKind.R8:
                    Show((R8)value);
                    break;
                case ExprTypeKind.BL:
                    Show((BL)value);
                    break;
                case ExprTypeKind.TX:
                    Show((TX)value);
                    break;
                default:
                    Contracts.Assert(false, "Unknown type");
                    break;
            }
        }

        private void Show(I4 x)
        {
            _wrt.Write(x);
        }

        private void Show(I8 x)
        {
            _wrt.Write(x);
        }

        private void Show(R4 x)
        {
            if (R4.IsNaN(x))
                _wrt.Write("NA");
            else
                _wrt.Write("{0:R}", x);
        }

        private void Show(R8 x)
        {
            if (R8.IsNaN(x))
                _wrt.Write("NA");
            else
                _wrt.Write("{0:R}", x);
        }

        private void Show(BL x)
        {
            if (!x)
                _wrt.Write("false");
            else
                _wrt.Write("true");
        }

        private void Show(TX str)
        {
            int len = str.Length;
            if (len > 100)
                len = 97; // Leave room for ...

            _wrt.Write('"');
            foreach (var ch in str.Span)
            {
                // Replace problematic characters with space.
                // REVIEW: Which characters should we replace?
                if (ch < ' ' || ch == '"')
                    _wrt.Write(' ');
                else
                    _wrt.Write(ch);
            }
            if (len < str.Length)
                _wrt.Write("...");
            _wrt.Write('"');
        }

        private void ShowType(ExprNode node)
        {
            if (!_showTypes)
                return;
            if (node.IsNone)
                return;

            _wrt.Write(':');
            _wrt.Write(node.ExprType.ToString());
        }

        private void ShowType(ParamNode node)
        {
            if (!_showTypes)
                return;
            if (node.ExprType == ExprTypeKind.None)
                return;

            _wrt.Write(':');
            _wrt.Write(node.ExprType.ToString());
        }

        public override void Visit(BoolLitNode node)
        {
            Contracts.AssertValue(node);
            _wrt.Write(node.Value ? "true" : "false");
            ShowType(node);
        }

        public override void Visit(StrLitNode node)
        {
            Contracts.AssertValue(node);
            Show(node.Value);
            ShowType(node);
        }

        public override void Visit(NumLitNode node)
        {
            Contracts.AssertValue(node);
            _wrt.Write(node.Value.ToString());
            ShowType(node);
        }

        public override void Visit(NameNode node)
        {
            Contracts.AssertValue(node);
            _wrt.Write(node.Value);
        }

        public override void Visit(IdentNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;
            _wrt.Write(node.Value);
            ShowType(node);
        }

        public override void Visit(ParamNode node)
        {
            Contracts.AssertValue(node);
            _wrt.Write(node.Name);
            ShowType(node);
        }

        public override void Visit(LambdaNode node)
        {
            Contracts.AssertValue(node);
            if (node.Vars.Length == 1)
                node.Vars[0].Accept(this);
            else
            {
                _wrt.Write('(');
                var pre = "";
                foreach (var v in node.Vars)
                {
                    _wrt.Write(pre);
                    v.Accept(this);
                    pre = "";
                }
                _wrt.Write(")");
            }
            _wrt.Write(" => ");
            node.Expr.Accept(this);
        }

        public override void Visit(UnaryOpNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            Precedence prec = GetPrec(node.Arg);
            _wrt.Write(GetString(node.Op));
            if (prec < Precedence.PrefixUnary)
                _wrt.Write('(');
            node.Arg.Accept(this);
            if (prec < Precedence.PrefixUnary)
                _wrt.Write(')');
            ShowType(node);
        }

        public override void Visit(BinaryOpNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            Precedence prec = GetPrec(node.Op);
            Precedence prec1 = GetPrec(node.Left);
            Precedence prec2 = GetPrec(node.Right);
            bool parens1 = NeedParensLeft(prec1, prec);
            bool parens2 = NeedParensRight(prec, prec2);

            if (parens1)
                _wrt.Write('(');
            node.Left.Accept(this);
            if (parens1)
                _wrt.Write(')');

            _wrt.Write(GetString(node.Op));

            if (parens2)
                _wrt.Write('(');
            node.Right.Accept(this);
            if (parens2)
                _wrt.Write(')');

            ShowType(node);
        }

        public override void Visit(ConditionalNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            Precedence prec0 = GetPrec(node.Cond);
            Precedence prec1 = GetPrec(node.Left);
            Precedence prec2 = GetPrec(node.Right);
            bool parens0 = NeedParensLeft(prec0, Precedence.Conditional);

            if (parens0)
                _wrt.Write('(');
            node.Cond.Accept(this);
            if (parens0)
                _wrt.Write(')');

            _wrt.Write(" ? ");
            node.Left.Accept(this);
            _wrt.Write(" : ");
            node.Right.Accept(this);

            ShowType(node);
        }

        public override void Visit(CompareNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            TokKind tidLax = node.TidLax;
            TokKind tidStrict = node.TidStrict;
            string strLax = GetString(tidLax);
            string strStrict = GetString(tidStrict);

            string str = string.Empty;
            string strOp = string.Empty;
            for (int i = 0; ;)
            {
                _wrt.Write(strOp);
                var arg = node.Operands.Items[i];
                var prec = GetPrec(arg);
                if (prec <= Precedence.Compare)
                    _wrt.Write('(');
                arg.Accept(this);
                if (prec <= Precedence.Compare)
                    _wrt.Write(')');
                if (++i >= node.Operands.Items.Length)
                    break;
                var tid = node.Operands.Delimiters[i - 1].Kind;
                Contracts.Assert(tid == tidLax || tid == tidStrict);
                strOp = tid == tidLax ? strLax : strStrict;
            }

            ShowType(node);
        }

        public override void Visit(CallNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            if (node.NameSpace != null)
            {
                node.NameSpace.Accept(this);
                _wrt.Write('.');
            }
            node.Head.Accept(this);
            _wrt.Write('(');
            node.Args.Accept(this);
            _wrt.Write(')');
            ShowType(node);
        }

        public override void Visit(ListNode node)
        {
            Contracts.AssertValue(node);

            int count = node.Items.Length;
            if (count == 0)
                return;

            if (node.Delimiters == null)
            {
                foreach (var child in node.Items)
                    child.Accept(this);
            }
            else if (count <= 6)
            {
                node.Items[0].Accept(this);
                for (int i = 1; i < count; i++)
                {
                    _wrt.Write(", ");
                    node.Items[i].Accept(this);
                }
            }
            else
            {
                for (int i = 0; i < 5; i++)
                {
                    node.Items[i].Accept(this);
                    _wrt.Write(", ");
                }
                _wrt.Write("..., ");
                node.Items[count - 1].Accept(this);
            }
        }

        public override void Visit(WithNode node)
        {
            Contracts.AssertValue(node);

            if (TryShowValue(node))
                return;

            _wrt.Write("with(");
            node.Local.Accept(this);
            _wrt.Write("; ");
            node.Body.Accept(this);
            _wrt.Write(")");

            ShowType(node);
        }

        public override void Visit(WithLocalNode node)
        {
            Contracts.AssertValue(node);

            _wrt.Write(node.Name);
            _wrt.Write(" = ");
            node.Value.Accept(this);
        }
    }
}
