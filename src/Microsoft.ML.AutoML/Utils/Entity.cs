// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace Microsoft.ML.AutoML
{
    internal static class EntityExtension
    {
        /// <summary>
        /// simplify entity by removing parenthesis and combine terms.
        /// for example
        /// (a + b) * (c + d) => a * c + a * d + b * c + b * d.
        /// </summary>
        /// <param name="entity">entity to be symplified</param>
        /// <returns>symplified entity.</returns>
        public static Entity Simplify(this Entity entity)
        {
            if (entity is ValueEntity || entity == null)
            {
                return entity;
            }

            var left = entity.Left.Simplify();
            var right = entity.Right.Simplify();

            if (entity is OneOfEntity)
            {
                return left + right;
            }
            else if (entity is ConcatenateEntity)
            {
                if (left is OneOfEntity && right is OneOfEntity)
                {
                    return (left.Left * right.Left + left.Left * right.Right + left.Right * right.Left + left.Right * right.Right).Simplify();
                }
                else if (left is OneOfEntity)
                {
                    return (left.Left * right + left.Right * right).Simplify();
                }
                else if (right is OneOfEntity)
                {
                    return (left * right.Left + left * right.Right).Simplify();
                }
                else
                {
                    return left * right;
                }
            }
            else
            {
                throw new Exception("Not Implemented");
            }
        }

        /// <summary>
        /// symplify entity into polynomial and split into terms.
        /// for example
        /// (a + b)*(c + d) => [a*c, a*d, b*c, b*d].
        /// </summary>
        /// <param name="entity"></param>
        /// <returns></returns>
        public static IEnumerable<Entity> ToTerms(this Entity entity)
        {
            var simplifiedEntity = entity.Simplify();
            if (simplifiedEntity is ValueEntity || simplifiedEntity is null || simplifiedEntity is ConcatenateEntity)
            {
                return new[] { simplifiedEntity };
            }
            else
            {
                var leftTerms = simplifiedEntity.Left.ToTerms();
                var rightTerms = simplifiedEntity.Right.ToTerms();
                return leftTerms.Concat(rightTerms);
            }
        }

        public static IEnumerable<ValueEntity> ValueEntities(this Entity entity)
        {
            if (entity is null)
            {
                return new List<ValueEntity>();
            }

            if (entity is ValueEntity)
            {
                return new[] { entity as ValueEntity };
            }
            else
            {
                return entity.Left.ValueEntities().Concat(entity.Right.ValueEntities());
            }
        }
    }

    public abstract class Entity
    {
        public Entity()
        {
        }

        public static Entity FromExpression(string expression)
        {
            var exp = SyntaxFactory.ParseExpression(expression);
            return Entity.FromExpression(exp);
        }

        public static Entity operator +(Entity left, Entity right)
        {
            var entity = new OneOfEntity();
            entity.Left = left;
            entity.Right = right;
            return entity;
        }

        public static Entity operator *(Entity left, Entity right)
        {
            var entity = new ConcatenateEntity();
            entity.Left = left;
            entity.Right = right;
            return entity;
        }

        public Entity Left { get; set; }

        public Entity Right { get; set; }

        private static Entity FromExpression(ExpressionSyntax exp)
        {
            if (exp is IdentifierNameSyntax i)
            {
                return new StringEntity(i.Identifier.ValueText);
            }
            else if (exp is ParenthesizedExpressionSyntax p)
            {
                return Entity.FromExpression(p.Expression);
            }
            else if (exp is BinaryExpressionSyntax b)
            {
                var left = Entity.FromExpression(b.Left);
                var right = Entity.FromExpression(b.Right);

                if (exp.Kind() == SyntaxKind.AddExpression)
                {
                    return new OneOfEntity()
                    {
                        Left = left,
                        Right = right,
                    };
                }
                else if (exp.Kind() == SyntaxKind.MultiplyExpression)
                {
                    return new ConcatenateEntity()
                    {
                        Left = left,
                        Right = right,
                    };
                }
                else
                {
                    throw new NotImplementedException();
                }
            }

            throw new NotImplementedException();
        }
    }

    /// <summary>
    /// concatenate one pipe behind another, aka '*'.
    /// </summary>
    internal class ConcatenateEntity : Entity
    {
        public override string ToString()
        {
            var left = Left.ToString();
            var right = Right.ToString();

            if (Left is OneOfEntity)
            {
                left = $"({left})";
            }

            if (Right is OneOfEntity)
            {
                right = $"({right})";
            }

            return $"{left} * {right}";
        }
    }

    /// <summary>
    /// select one of the pipe among its children, aka '+'.
    /// </summary>
    internal class OneOfEntity : Entity
    {
        public override string ToString()
        {
            var left = Left.ToString();
            var right = Right.ToString();

            return $"{left} + {right}";
        }
    }

    internal abstract class ValueEntity : Entity
    {
    }

    internal class StringEntity : ValueEntity
    {
        public StringEntity(string value)
        {
            Value = value;
        }

        public string Value { get; }

        public override string ToString()
        {
            return Value;
        }
    }

    internal class EstimatorEntity : ValueEntity
    {
        public EstimatorEntity(Estimator estimator)
        {
            Estimator = estimator;
        }

        public Estimator Estimator { get; }
    }
}
