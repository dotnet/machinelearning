# [ML.NET](http://dot.net/ml) 3.0.1

## **New Features**
- **Add support for Apache.Arrow.Types.TimestampType to DataFrame** ([#6871](https://github.com/dotnet/machinelearning/pull/6871)) - Thanks @asmirnov82!


## **Enhancements**
- **Update TorchSharp to latest version** ([#6954](https://github.com/dotnet/machinelearning/pull/6954))
- **Reorganize dataframe files** ([#6872](https://github.com/dotnet/machinelearning/pull/6872)) - Thanks @asmirnov82!
- **Add sample variance and standard deviation to NormalizeMeanVariance** ([#6885](https://github.com/dotnet/machinelearning/pull/6885)) - Thanks @tearlant!
- **Fixes NER to correctly expand/shrink the labels** ([#6928](https://github.com/dotnet/machinelearning/pull/6928))


## **Bug Fixes**
- **Fix SearchSpace reference not being included** ([#6951](https://github.com/dotnet/machinelearning/pull/6951))
- **Rename NameEntity to NamedEntity** ([#6917](https://github.com/dotnet/machinelearning/pull/6917))
- **Fix assert by only accessing idx** ([#6924](https://github.com/dotnet/machinelearning/pull/6924))


## **Build / Test updates**
- **Add Backport github workflow** ([#6944](https://github.com/dotnet/machinelearning/pull/6944))
- **Branding for 3.0.1** ([#6943](https://github.com/dotnet/machinelearning/pull/6943))
- **Only use semi-colons for NoWarn - fixes build break** ([#6935](https://github.com/dotnet/machinelearning/pull/6935))
- **Update dependencies from dotnet/arcade** ([#6703](https://github.com/dotnet/machinelearning/pull/6703))
- **Update dependencies from dotnet/arcade** ([#6957](https://github.com/dotnet/machinelearning/pull/6957))
- **Migrate to the 'locker' GitHub action for locking closed/stale issues/PRs** ([#6896](https://github.com/dotnet/machinelearning/pull/6896))
- **Make double assertions compare with tolerance instead of precision** ([#6923](https://github.com/dotnet/machinelearning/pull/6923))
- **Don't include the SDK in our helix payload** ([#6918](https://github.com/dotnet/machinelearning/pull/6918))


## **Documentation Updates**
- **Updated ml.net versioning** ([#6907](https://github.com/dotnet/machinelearning/pull/6907))
- **Update developer-guide.md** ([#6870](https://github.com/dotnet/machinelearning/pull/6870)) - Thanks @computerscienceiscool!
- **Update release-3.0.0.md** ([#6895](https://github.com/dotnet/machinelearning/pull/6895)) - Thanks @taeerhebend!


## **Breaking changes**
- **Rename NameEntity to NamedEntity** ([#6917](https://github.com/dotnet/machinelearning/pull/6917))
