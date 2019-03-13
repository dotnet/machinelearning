﻿using System;
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
            IHostEnvironment env = null;
            var text = TextLoaderStatic.CreateLoader(env, ctx => (
                label: ctx.LoadBool(0),
                text: ctx.LoadText(1),
                numericFeatures: ctx.LoadFloat(2, 5)));

            var est = text.MakeNewEstimator();
            // This should work.
            est.Append(r => r.text);
            // These should not.
            est.Append(r => 5);
            est.Append(r => new { r.text, bad = 2 });
            // This should work.
            est.Append(r => Tuple.Create(r.text, r.numericFeatures));
            // This should work.
            est.Append(r => (a: r.text, b: r.label, c: (d: r.text, r.label)));
            // This should not, and it should indicate a path to the problematic item.
            est.Append(r => (a: r.text, b: r.label, c: (d: r.text, 5.2f)));

            // Check a different entrance into static land now, with one of the asserts.
            var view = text.Load(null).AsDynamic;
            // Despite the fact that the names are all wrong, this should still work
            // from the point of view of this analyzer.
            view.AssertStatic(env, c => (
               stay: c.KeyU4.TextValues.Scalar,
               awhile: c.KeyU1.I4Values.Vector));
            // However, this should not.
            view.AssertStatic(env, c => (
               and: c.KeyU4.TextValues.Scalar,
               listen: 1l));
        }
    }
}
