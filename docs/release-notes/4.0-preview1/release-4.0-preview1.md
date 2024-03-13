# [ML.NET](http://dot.net/ml) 4.0-preview1

## **New Features**
- **Add sweepable estimator to NER** ([6965](https://github.com/dotnet/machinelearning/pull/6965))
- **Introducing Tiktoken Tokenizer** ([6981](https://github.com/dotnet/machinelearning/pull/6981))
- **Add text normalizer transformer to AutoML** ([6998](https://github.com/dotnet/machinelearning/pull/6998))

## **Enhancements**
- **Add support for Apache.Arrow.Types.TimestampType to DataFrame** ([6871](https://github.com/dotnet/machinelearning/pull/6871)) - Thanks @asmirnov82!
- **Add new type to key-value converter** ([6973](https://github.com/dotnet/machinelearning/pull/6973))
- **Update OnnxRuntime to 1.16.3** ([6975](https://github.com/dotnet/machinelearning/pull/6975))
- **Tokenizer's Interfaces Cleanup** ([7001](https://github.com/dotnet/machinelearning/pull/7001))
- **Match  SweepableEstimatorFactory name with Ml.net name.** ([7007](https://github.com/dotnet/machinelearning/pull/7007))
- **First round of perf improvements for tiktoken** ([7012](https://github.com/dotnet/machinelearning/pull/7012))
- **Tweak CreateByModelNameAsync** ([7015](https://github.com/dotnet/machinelearning/pull/7015))
- **Avoid LruCache in Tiktoken when cacheSize specified is 0** ([7016](https://github.com/dotnet/machinelearning/pull/7016))
- **Tweak Tiktoken's BytePairEncode for improved perf** ([7017](https://github.com/dotnet/machinelearning/pull/7017))
- **Optimize regexes used in tiktoken** ([7020](https://github.com/dotnet/machinelearning/pull/7020))
- **Address the feedback on the tokenizer's library** ([7024](https://github.com/dotnet/machinelearning/pull/7024))
- **Add Span support in tokenizer's Model abstraction** ([7035](https://github.com/dotnet/machinelearning/pull/7035))
- **Adding needed Tokenizer's APIs** ([7047](https://github.com/dotnet/machinelearning/pull/7047))

## **Bug Fixes**
- **Fix formatting that fails in VS** ([7023](https://github.com/dotnet/machinelearning/pull/7023))
- **Issue #6606 - Add sample variance and standard deviation to NormalizeMeanVariance** ([6885](https://github.com/dotnet/machinelearning/pull/6885)) - Thanks @tearlant!

## **Build / Test updates**
- **Migrate to the 'locker' GitHub action for locking closed/stale issues/PRs** ([6896](https://github.com/dotnet/machinelearning/pull/6896))
- **Reorganize dataframe files** ([6872](https://github.com/dotnet/machinelearning/pull/6872)) - Thanks @asmirnov82!
- **Updated ml.net versioning** ([6907](https://github.com/dotnet/machinelearning/pull/6907))
- **Don't include the SDK in our helix payload** ([6918](https://github.com/dotnet/machinelearning/pull/6918))
- **Make double assertions compare with tolerance instead of precision** ([6923](https://github.com/dotnet/machinelearning/pull/6923))
- **Fix assert by only accessing idx** ([6924](https://github.com/dotnet/machinelearning/pull/6924))
- **Only use semi-colons for NoWarn - fixes build break** ([6935](https://github.com/dotnet/machinelearning/pull/6935))
- **Packaging cleanup** ([6939](https://github.com/dotnet/machinelearning/pull/6939))
- **Add Backport github workflow** ([6944](https://github.com/dotnet/machinelearning/pull/6944))
- **[main] Update dependencies from dotnet/arcade** ([6957](https://github.com/dotnet/machinelearning/pull/6957))
- **Update .NET Runtimes to latest version** ([6964](https://github.com/dotnet/machinelearning/pull/6964))
- **Testing light gbm bad allocation** ([6968](https://github.com/dotnet/machinelearning/pull/6968))
- **[main] Update dependencies from dotnet/arcade** ([6969](https://github.com/dotnet/machinelearning/pull/6969))
- **[main] Update dependencies from dotnet/arcade** ([6976](https://github.com/dotnet/machinelearning/pull/6976))
- **FabricBot: Onboarding to GitOps.ResourceManagement because of FabricBot decommissioning** ([6983](https://github.com/dotnet/machinelearning/pull/6983))
- **[main] Update dependencies from dotnet/arcade** ([6985](https://github.com/dotnet/machinelearning/pull/6985))
- **[main] Update dependencies from dotnet/arcade** ([6995](https://github.com/dotnet/machinelearning/pull/6995))
- **Temp fix for the race condition during the tests.** ([7021](https://github.com/dotnet/machinelearning/pull/7021))
- **Make MlImage tests not block file for reading** ([7029](https://github.com/dotnet/machinelearning/pull/7029))
- **Remove SourceLink SDK references** ([7037](https://github.com/dotnet/machinelearning/pull/7037))
- **Change official build to use 1ES templates** ([7048](https://github.com/dotnet/machinelearning/pull/7048))
- **Auto-generated baselines by 1ES Pipeline Templates** ([7051](https://github.com/dotnet/machinelearning/pull/7051))
- **Update package versions in use by ML.NET tests** ([7055](https://github.com/dotnet/machinelearning/pull/7055))
- **testing arm python brew overwite** ([7058](https://github.com/dotnet/machinelearning/pull/7058))

## **Documentation Updates**
- **Update developer-guide.md** ([6870](https://github.com/dotnet/machinelearning/pull/6870)) - Thanks @computerscienceiscool!
- **Update release-3.0.0.md** ([6895](https://github.com/dotnet/machinelearning/pull/6895)) - Thanks @taeerhebend!
