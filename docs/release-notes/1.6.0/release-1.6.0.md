# [ML.NET](http://dot.net/ml) 1.6.0

## **New Features**
- **Support for Arm/Arm64/Applie Silicon has been added**. ([#5789](https://github.com/dotnet/machinelearning/pull/5789)) You can now use most ML.NET on Arm/Arm64/Apple Silicon devices. Anything without a hard depdency on x86 SIMD instructions or Intel MKL are supported. 
- **Support for specifying a temp path ML.NET will use**. ([#5782](https://github.com/dotnet/machinelearning/pull/5782)) You can now set the TempFilePath in the MLContext that it will use.
- **Support for specifying the recursion limit to use when loading an ONNX model** ([#5840](https://github.com/dotnet/machinelearning/pull/5840)) The recursion limit defaults to 100, but you can now specify the value in case you need to use a larger number. (**Thanks @Crabzmatic**)
- **Support for saving Tensorflow models in the SavedModel format added** ([#5797](https://github.com/dotnet/machinelearning/pull/5797)) You can now save models that use the Tensorflow SavedModel format instead of just the frozen graph format. (**Thanks @darth-vader-lg**)


## **Enhancements**
- Switched from using a fork of SharpZipLib to using the official package ([#5735](https://github.com/dotnet/machinelearning/pull/5735))
- Let user specify a temp path location ([#5782](https://github.com/dotnet/machinelearning/pull/5782))
- Clean up onnx temp models by opening with a "Delete on close" flag ([#5782](https://github.com/dotnet/machinelearning/pull/5782))
- Ensures the named model is loaded in a PredictionEnginePool before use ([#5833](https://github.com/dotnet/machinelearning/pull/5833)) (**Thanks @feiyun0112**)
- Use indentation for 'if' ([#5825](https://github.com/dotnet/machinelearning/pull/5825)) (**Thanks @feiyun0112**)
- Use Append instead of AppendFormat if we don't need formatting ([#5826](https://github.com/dotnet/machinelearning/pull/5826)) (**Thanks @feiyun0112**)
- Cast by using is operator ([#5829](https://github.com/dotnet/machinelearning/pull/5829)) (**Thanks @feiyun0112**)
- Removed un-neccesary return statements ([#5828](https://github.com/dotnet/machinelearning/pull/5828)) (**Thanks @feiyun0112**)
- Removed code that could never be executed ([#5808](https://github.com/dotnet/machinelearning/pull/5808)) (**Thanks @feiyun0112**)
- Remove some empty statements ([#5827](https://github.com/dotnet/machinelearning/pull/5827)) (**Thanks @feiyun0112**)
- Added in short-circuit logic for conditionals ([#5824](https://github.com/dotnet/machinelearning/pull/5824)) (**Thanks @feiyun0112**)
- Update LightGBM to v2.3.1 ([#5851](https://github.com/dotnet/machinelearning/pull/5851))
- Raised the default recursion limit for ONNX models from 10 to 100. ([#5796](https://github.com/dotnet/machinelearning/pull/5796)) (**Thanks @darth-vader-lg**)
- Speed up the inference of the Tensorflow saved_models. ([#5848](https://github.com/dotnet/machinelearning/pull/5848)) (**Thanks @darth-vader-lg**)
- Speed-up bitmap operations on images. ([#5857](https://github.com/dotnet/machinelearning/pull/5857)) (**Thanks @darth-vader-lg**)
- Updated to latest version of Intel MKL. ([#5867](https://github.com/dotnet/machinelearning/pull/5867))
- **AutoML.NET specific enhancements**
- Offer suggestions for possibly mistyped label column names in AutoML ([#5624](https://github.com/dotnet/machinelearning/pull/5624)) (**Thanks @Crabzmatic**)


## **Bug Fixes**
- Fix erroneous use of TaskContinuationOptions in ThreadUtils.cs ([#5753](https://github.com/dotnet/machinelearning/pull/5753))
- Fix a few locations that can try to access a null object ([#5804](https://github.com/dotnet/machinelearning/pull/5804)) (**Thanks @feiyun0112**)
- Use return value of method ([#5818](https://github.com/dotnet/machinelearning/pull/5818)) (**Thanks @feiyun0112**)
- Adding throw to some exceptions that weren't throwing them originally ([#5823](https://github.com/dotnet/machinelearning/pull/5823)) (**Thanks @feiyun0112**)
- Fixed a situation in the CountTargetEncodingTransformer where it never reached the stop condition ([#5822](https://github.com/dotnet/machinelearning/pull/5822)) (**Thanks @feiyun0112**)


## **Build / Test updates**
- Changed default branch from master to main ([#5715](https://github.com/dotnet/machinelearning/pull/5715)) ([#5717](https://github.com/dotnet/machinelearning/pull/5717)) ([#5719](https://github.com/dotnet/machinelearning/pull/5719))
- Fix for libomp in the CI process for MacOS 11 ([#5771](https://github.com/dotnet/machinelearning/pull/5771))
- Minor code cleanup. ([#5770](https://github.com/dotnet/machinelearning/pull/5770))
- Updated arcade to the latest version ([#5783](https://github.com/dotnet/machinelearning/pull/5783))
- Switched signing certificate to use dotnet certificate ([#5794](https://github.com/dotnet/machinelearning/pull/5794))
- Building natively and cross targeting for Arm/Arm64/Apple Silicon is now supported. ([#5789](https://github.com/dotnet/machinelearning/pull/5789))
- Upload classic pdb to symweb ([#5816](https://github.com/dotnet/machinelearning/pull/5816))
- Fix MacoOS CI issue ([#5854](https://github.com/dotnet/machinelearning/pull/5854))
- Added in a Helix Integration for testing. ([#5837](https://github.com/dotnet/machinelearning/pull/5837))


## **Documentation Updates**
- Fixed markdown issues in MulticlassClassificationMetris and CalibratedBinaryClassificationMetrics ([#5732](https://github.com/dotnet/machinelearning/pull/5732)) (**Thanks @R0Wi**)
- Update unix instructions for x-compiling on ARM ([#5811](https://github.com/dotnet/machinelearning/pull/5811))
- Update Contribution.MD with description of help wanted tags ([#5815](https://github.com/dotnet/machinelearning/pull/5815))
- Add Korean translation for repo readme.md ([#5780](https://github.com/dotnet/machinelearning/pull/5780)) (**Thanks @metr0jw**)
- Fix spelling error in MLContext class summary ([#5832](https://github.com/dotnet/machinelearning/pull/5832)) (**Thanks @Crabzmatic**)
- Update issue templates ([#5846](https://github.com/dotnet/machinelearning/pull/5846))

## **Breaking Changes**
- None
