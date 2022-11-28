# Interfacing XGBoost with ML.NET

The design of the XGBoost integration follows as closely as possible
that of LightGBM's.  XGBoost and LightGBM have substantial
differences, and, collectively, notable differences with other
algorithms.  Most obviously, both XGBoost and LightGBM include a
substantial surface of configurable options.  Even what ML task
(regression, classification, ranking) is to be carried out is
specified by a configuration knob.  This impacts the design of the
integration, that factorizes commonalities in a base class and leaves
task-specific options in a subclass.

Compared to other `Estimator`/`Transformer` pairs, there are three
"subsystems" that have unique characteristics:
* handling of options
* loading of training/validation data
* use of `ValueMapper` predictor

## Configuration

A specific design decision was induced by whether the option names
should respect the LightGBM terminology (so that users can see an
as-close-as-possible interface between the two integrations in
ML.NET), or privilege XGBoost nomenclature.  The latter course of
action was taken, with the rationale that those options are well
documented (arguably, better than LightGBM's) through integrations of
XGBoost in other languages (most notably, Python, R and Java).


