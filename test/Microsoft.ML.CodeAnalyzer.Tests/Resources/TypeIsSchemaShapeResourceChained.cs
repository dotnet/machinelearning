using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.CommandLine;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.StaticPipe;

namespace Bubba
{
    class Foo
    {
        public static void Bar()
        {
            DataLoader<IMultiStreamSource, T> Foo1<T>(Func<TextLoaderStatic.Context, T> m)
            {
                IHostEnvironment env = null;
                // We ought to fail here.
                return TextLoaderStatic.CreateLoader(env, m);
            }

            DataLoader<IMultiStreamSource, T> Foo2<[IsShape] T>(Func<TextLoaderStatic.Context, T> m)
            {
                IHostEnvironment env = null;
                // We ought not to fail here due to that [IsShape], but calls to this method might fail.
                return TextLoaderStatic.CreateLoader(env, m);
            }

            DataLoader<IMultiStreamSource, T> Foo3<T>(Func<TextLoaderStatic.Context, T> m)
                where T : PipelineColumn
            {
                IHostEnvironment env = null;
                // This should work.
                return TextLoaderStatic.CreateLoader(env, m);
            }

            DataLoader<IMultiStreamSource, T> Foo4<T>(Func<TextLoaderStatic.Context, T> m)
                where T : IEnumerable<int>
            {
                IHostEnvironment env = null;
                // This should not work.
                return TextLoaderStatic.CreateLoader(env, m);
            }

            void Scratch()
            {
                // Neither of these two should fail here, though the method they're calling ought to fail.
                var f1 = Foo1(ctx => (
                    label: ctx.LoadBool(0), text: ctx.LoadText(1)));
                var f2 = Foo1(ctx => (
                    label: ctx.LoadBool(0), text: 5));

                // The first should succeed, the second should fail.
                var f3 = Foo2(ctx => (
                    label: ctx.LoadBool(0), text: ctx.LoadText(1)));
                var f4 = Foo2(ctx => (
                    label: ctx.LoadBool(0), text: 6));

                // This should succeed.
                var f5 = Foo3(ctx => ctx.LoadBool(0));
            }
        }
    }
}
