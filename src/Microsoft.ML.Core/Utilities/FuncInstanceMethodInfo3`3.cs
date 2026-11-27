// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#nullable enable

using System;
using System.Linq.Expressions;
using System.Reflection;
using Microsoft.ML.Runtime;

namespace Microsoft.ML.Internal.Utilities
{
    /// <summary>
    /// Represents the <see cref="MethodInfo"/> for a generic function corresponding to <see cref="Func{T, TResult}"/>,
    /// with the following characteristics:
    ///
    /// <list type="bullet">
    /// <item><description>The method is an instance method on an object of type <typeparamref name="TTarget"/>.</description></item>
    /// <item><description>Three generic type arguments.</description></item>
    /// <item><description>A return value of <typeparamref name="TResult"/>.</description></item>
    /// </list>
    /// </summary>
    /// <typeparam name="TTarget">The type of the receiver of the instance method.</typeparam>
    /// <typeparam name="T">The type of the parameter of the method.</typeparam>
    /// <typeparam name="TResult">The type of the return value of the method.</typeparam>
    internal sealed class FuncInstanceMethodInfo3<TTarget, T, TResult> : FuncMethodInfo3<T, TResult>
        where TTarget : class
    {
        private static readonly string _targetTypeCheckMessage = $"Should have a target type of '{typeof(TTarget)}'";

        public FuncInstanceMethodInfo3(Func<T, TResult> function)
            : this(function.Method)
        {
        }

        private FuncInstanceMethodInfo3(MethodInfo methodInfo)
            : base(methodInfo)
        {
            Contracts.CheckParam(!GenericMethodDefinition.IsStatic, nameof(methodInfo), Strings.ShouldBeAnInstanceMethod);
            Contracts.CheckParam(GenericMethodDefinition.DeclaringType == typeof(TTarget), nameof(methodInfo), _targetTypeCheckMessage);
        }

        /// <summary>
        /// Creates a <see cref="FuncInstanceMethodInfo1{TTarget, T, TResult}"/> representing the <see cref="MethodInfo"/>
        /// for a generic instance method. This helper method allows the instance to be created prior to the creation of
        /// any instances of the target type. The following example shows the creation of an instance representing the
        /// <see cref="object.Equals(object)"/> method:
        ///
        /// <code>
        /// FuncInstanceMethodInfo1&lt;object, object, int&gt;.Create(obj => obj.Equals)
        /// </code>
        /// </summary>
        /// <param name="expression">The expression which creates the delegate for an instance of the target type.</param>
        /// <returns>A <see cref="FuncInstanceMethodInfo1{TTarget, T, TResult}"/> representing the <see cref="MethodInfo"/>
        /// for the generic instance method.</returns>
        public static FuncInstanceMethodInfo3<TTarget, T, TResult> Create(Expression<Func<TTarget, Func<T, TResult>>> expression)
        {
            if (!(expression is { Body: UnaryExpression { Operand: MethodCallExpression methodCallExpression } }))
            {
                throw Contracts.ExceptParam(nameof(expression), Strings.UnexpectedExpressionForm);
            }

            // Verify that we are calling MethodInfo.CreateDelegate(Type, object)
            Contracts.CheckParam(methodCallExpression.Method.DeclaringType == typeof(MethodInfo), nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Method.Name == nameof(MethodInfo.CreateDelegate), nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Method.GetParameters().Length == 2, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Method.GetParameters()[0].ParameterType == typeof(Type), nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Method.GetParameters()[1].ParameterType == typeof(object), nameof(expression), Strings.UnexpectedExpressionForm);

            // Verify that we are creating a delegate of type Func<T, TResult>
            Contracts.CheckParam(methodCallExpression.Arguments.Count == 2, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Arguments[0] is ConstantExpression, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(((ConstantExpression)methodCallExpression.Arguments[0]).Type == typeof(Type), nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam((Type)((ConstantExpression)methodCallExpression.Arguments[0]).Value == typeof(Func<T, TResult>), nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Arguments[1] is ParameterExpression, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(methodCallExpression.Arguments[1] == expression.Parameters[0], nameof(expression), Strings.UnexpectedExpressionForm);

            // Check the MethodInfo
            Contracts.CheckParam(methodCallExpression.Object is ConstantExpression, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(((ConstantExpression)methodCallExpression.Object).Type == typeof(MethodInfo), nameof(expression), Strings.UnexpectedExpressionForm);

            var methodInfo = (MethodInfo)((ConstantExpression)methodCallExpression.Object).Value;
            Contracts.CheckParam(expression.Body is UnaryExpression, nameof(expression), Strings.UnexpectedExpressionForm);
            Contracts.CheckParam(((UnaryExpression)expression.Body).Operand is MethodCallExpression, nameof(expression), Strings.UnexpectedExpressionForm);

            return new FuncInstanceMethodInfo3<TTarget, T, TResult>(methodInfo);
        }
    }
}
