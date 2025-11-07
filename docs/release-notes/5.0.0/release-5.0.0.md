# [ML.NET](http://dot.net/ml) 5.0.0

## **New Features**
- **[GenAI] Introduce CausalLMPipelineChatClient for MEAI.IChatClient** ([#7270](https://github.com/dotnet/machinelearning/pull/7270))
- **Introducing SentencePiece Unigram Tokenizer Model** ([#7390](https://github.com/dotnet/machinelearning/pull/7390))
- **Phi-4 Tokenizer Support** ([#7396](https://github.com/dotnet/machinelearning/pull/7396))
- **Support O3 OpenAI model mapping** ([#7394](https://github.com/dotnet/machinelearning/pull/7394))
- **Support ByteLevel encoding in Bpe tokenizer to support DeepSeek model** ([#7425](https://github.com/dotnet/machinelearning/pull/7425))
- **Support Tiktoken Gpt-4.1 Model** ([#7453](https://github.com/dotnet/machinelearning/pull/7453))
- **Support OpenAI OSS Models with Tiktoken tokenizer** ([#7494](https://github.com/dotnet/machinelearning/pull/7494))
- **Add deterministic option for LightGBM** ([#7415](https://github.com/dotnet/machinelearning/pull/7415))
- **Added NumberOfLeaves to FastForestRegression and FastForestOva options** ([#7499](https://github.com/dotnet/machinelearning/pull/7499)) - Thanks @JoshuaSloan!

## **Enhancements**
- **Add Timeout to Regex used in the tokenizers** ([#7284](https://github.com/dotnet/machinelearning/pull/7284))
- **Final tokenizer's cleanup** ([#7291](https://github.com/dotnet/machinelearning/pull/7291))
- **Update System.Numerics.Tensors version** ([#7322](https://github.com/dotnet/machinelearning/pull/7322)) - Thanks @asmirnov82!
- **[GenAI] SFT Example** ([#7316](https://github.com/dotnet/machinelearning/pull/7316))
- **Update M.E.AI version used by Microsoft.ML.GenAI.Core** ([#7329](https://github.com/dotnet/machinelearning/pull/7329))
- **Update DependencyModel** ([#7338](https://github.com/dotnet/machinelearning/pull/7338))
- **Some tweaks to the Microsoft.ML.Tokenizers PACKAGE.md** ([#7360](https://github.com/dotnet/machinelearning/pull/7360))
- **Consolidate System.Numerics.Tensors dependency** ([#7356](https://github.com/dotnet/machinelearning/pull/7356)) - Thanks @asmirnov82!
- **Update Microsoft.Extensions.AI to 9.3.0-preview.1.25114.11** ([#7388](https://github.com/dotnet/machinelearning/pull/7388))
- **Create SentencePieceTokenizer from options object** ([#7403](https://github.com/dotnet/machinelearning/pull/7403))
- **Unigram tokenizer fixes** ([#7409](https://github.com/dotnet/machinelearning/pull/7409))
- **Update to M.E.AI 9.3.0-preview.1.25161.3** ([#7414](https://github.com/dotnet/machinelearning/pull/7414))
- **Reduce usage of unsafe constructs throughout codebase** ([#7426](https://github.com/dotnet/machinelearning/pull/7426)) - Thanks @GrabYourPitchforks!
- **Cleanup SentencePiece tokenizer** ([#7427](https://github.com/dotnet/machinelearning/pull/7427))
- **Update to M.E.AI 9.4.0-preview.1.25207.5** ([#7439](https://github.com/dotnet/machinelearning/pull/7439))
- **Update to M.E.AI 9.4.3-preview.1.25230.7** ([#7459](https://github.com/dotnet/machinelearning/pull/7459))
- **Update to stable Microsoft.Extensions.AI.Abstractions** ([#7466](https://github.com/dotnet/machinelearning/pull/7466))
- **Convert repository to NuGet Central Package Management** ([#7482](https://github.com/dotnet/machinelearning/pull/7482))
- **Rename Casual to Causal** ([#7484](https://github.com/dotnet/machinelearning/pull/7484)) - Thanks @feiyun0112!
- **Updated Tensorflow.Net to 0.70.2 with Tensorflow 2.7.0.** ([#7472](https://github.com/dotnet/machinelearning/pull/7472)) - Thanks @Crichen!
- **Mark internal classes as internal** ([#7511](https://github.com/dotnet/machinelearning/pull/7511))
- **Address the design review feedback** ([#7513](https://github.com/dotnet/machinelearning/pull/7513))
- **BpeTokenizer Cleanup** ([#7514](https://github.com/dotnet/machinelearning/pull/7514))
- **Improve native build and mark our official build as CFS Clean** ([#7516](https://github.com/dotnet/machinelearning/pull/7516))
- **Improve unique directory generation for temp files** ([#7520](https://github.com/dotnet/machinelearning/pull/7520))
- **Updating OnnxRuntime** ([#7469](https://github.com/dotnet/machinelearning/pull/7469))

## **Bug Fixes**
- **Fix broken inheritance from DataFrameColumn class** ([#7324](https://github.com/dotnet/machinelearning/pull/7324)) - Thanks @asmirnov82!
- **Moved SpecialTokens assignment after the modification to avoid "Collection Modified" error** ([#7328](https://github.com/dotnet/machinelearning/pull/7328)) - Thanks @shaltielshmid!
- **Fix DateTime export to csv with culture info** ([#7358](https://github.com/dotnet/machinelearning/pull/7358)) - Thanks @asmirnov82!
- **Increase cancelling waiting time for AutoMLExperiment_return_current_best_trial_when_ct_is_canceled_with_trial_completed_Async** ([#7424](https://github.com/dotnet/machinelearning/pull/7424))
- **Fixed light gbm update** ([#7431](https://github.com/dotnet/machinelearning/pull/7431))
- **Fix incorrect IntPtr null check in FftUtils** ([#7434](https://github.com/dotnet/machinelearning/pull/7434)) - Thanks @GrabYourPitchforks!
- **ImageClassificationTrainer PredictedLabelColumnName bug when the name is not default** ([#7458](https://github.com/dotnet/machinelearning/pull/7458)) - Thanks @feiyun0112!
- **Fix ElementwiseGreaterThanOrEqual to use >= instead of ==** ([#7475](https://github.com/dotnet/machinelearning/pull/7475)) - Thanks @vsarakhan!
- **Fix minor typo in BinFinder.cs** ([#7496](https://github.com/dotnet/machinelearning/pull/7496)) - Thanks @KM5075!
- **Fix PositiveRecall optimization in AutoMLExperiment** ([#7493](https://github.com/dotnet/machinelearning/pull/7493)) - Thanks @JoshuaSloan!

## **Build / Test updates**
- **Add the components governance file `cgmanifest.json` for tokenizer's vocab files** ([#7283](https://github.com/dotnet/machinelearning/pull/7283))
- **Update To MacOS 13** ([#7285](https://github.com/dotnet/machinelearning/pull/7285))
- **Updated remote executor** ([#7295](https://github.com/dotnet/machinelearning/pull/7295))
- **Fixing native lookup** ([#7282](https://github.com/dotnet/machinelearning/pull/7282))
- **Update dependencies from maintenance-packages to latest versions** ([#7301](https://github.com/dotnet/machinelearning/pull/7301))
- **Maintenance package version updates.** ([#7304](https://github.com/dotnet/machinelearning/pull/7304))
- **Fixing tokenizers version** ([#7309](https://github.com/dotnet/machinelearning/pull/7309))
- **Update version for 5.0** ([#7311](https://github.com/dotnet/machinelearning/pull/7311))
- **Update dynamic loading report reference** ([#7321](https://github.com/dotnet/machinelearning/pull/7321)) - Thanks @emmanuel-ferdman!
- **Net8 tests** ([#7319](https://github.com/dotnet/machinelearning/pull/7319))
- **[main] Update dependencies from dotnet/arcade** ([#7266](https://github.com/dotnet/machinelearning/pull/7266))
- **[main] Update dependencies from dotnet/arcade** ([#7352](https://github.com/dotnet/machinelearning/pull/7352))
- **Update MSTest to latest** ([#7349](https://github.com/dotnet/machinelearning/pull/7349)) - Thanks @Youssef1313!
- **[main] Update dependencies from dotnet/arcade** ([#7368](https://github.com/dotnet/machinelearning/pull/7368))
- **[main] Update dependencies from dotnet/arcade** ([#7374](https://github.com/dotnet/machinelearning/pull/7374))
- **[main] Update dependencies from dotnet/arcade** ([#7376](https://github.com/dotnet/machinelearning/pull/7376))
- **[main] Update dependencies from dotnet/arcade** ([#7382](https://github.com/dotnet/machinelearning/pull/7382))
- **[main] Update dependencies from dotnet/arcade** ([#7387](https://github.com/dotnet/machinelearning/pull/7387))
- **Update Helix ubuntu arm32 container** ([#7410](https://github.com/dotnet/machinelearning/pull/7410))
- **Update dependencies from maintenance-packages** ([#7412](https://github.com/dotnet/machinelearning/pull/7412))
- **[main] Update dependencies from dotnet/arcade** ([#7397](https://github.com/dotnet/machinelearning/pull/7397))
- **Switch to AwesomeAssertions** ([#7421](https://github.com/dotnet/machinelearning/pull/7421))
- **Update maintenance-dependencies** ([#7433](https://github.com/dotnet/machinelearning/pull/7433))
- **update cmake mac** ([#7443](https://github.com/dotnet/machinelearning/pull/7443))
- **[main] Update dependencies from dotnet/arcade** ([#7423](https://github.com/dotnet/machinelearning/pull/7423))
- **[main] Update dependencies from dotnet/arcade** ([#7455](https://github.com/dotnet/machinelearning/pull/7455))
- **Dependency version updates** ([#7457](https://github.com/dotnet/machinelearning/pull/7457))
- **[main] Update dependencies from dotnet/arcade** ([#7463](https://github.com/dotnet/machinelearning/pull/7463))
- **Create copilot-setup-steps.yml** ([#7478](https://github.com/dotnet/machinelearning/pull/7478))
- **Add copilot-setup-steps.yml** ([#7481](https://github.com/dotnet/machinelearning/pull/7481))
- **Enable dependabot.** ([#7486](https://github.com/dotnet/machinelearning/pull/7486))
- **macOS x64 CI: fix dependency install and OpenMP runtime copy (use Homebrew libomp, adjust Helix payload)** ([#7510](https://github.com/dotnet/machinelearning/pull/7510)) - Thanks @asp2286!
- **Initialize es-metadata.yml for inventory** ([#7504](https://github.com/dotnet/machinelearning/pull/7504))
- **Update Windows image, fix mac build** ([#7515](https://github.com/dotnet/machinelearning/pull/7515))
- **[main] Update dependencies from dotnet/arcade** ([#7473](https://github.com/dotnet/machinelearning/pull/7473))
- **[main] Update dependencies from dotnet/arcade** ([#7519](https://github.com/dotnet/machinelearning/pull/7519))
- **Remove baselines** ([#7526](https://github.com/dotnet/machinelearning/pull/7526))
- **[main] Update dependencies from dotnet/arcade** ([#7521](https://github.com/dotnet/machinelearning/pull/7521))
- **Use arcade script for installing MacOS dependencies** ([#7533](https://github.com/dotnet/machinelearning/pull/7533))
- **[main] Update dependencies from dotnet/arcade** ([#7532](https://github.com/dotnet/machinelearning/pull/7532))

## **Documentation Updates**
- **4.0 release notes** ([#7302](https://github.com/dotnet/machinelearning/pull/7302))
- **Fix up docs for MLContext** ([#7334](https://github.com/dotnet/machinelearning/pull/7334)) - Thanks @gewarren!
- **Added in 5.0 preview 1 release notes** ([#7400](https://github.com/dotnet/machinelearning/pull/7400))
- **[main] Added 4.0.2 servicing release notes** ([#7401](https://github.com/dotnet/machinelearning/pull/7401))
- **Updated preview release notes.** ([#7405](https://github.com/dotnet/machinelearning/pull/7405))
- **Update Tokenizer conceptual doc link in package docs** ([#7445](https://github.com/dotnet/machinelearning/pull/7445))
- **Random doc updates** ([#7476](https://github.com/dotnet/machinelearning/pull/7476)) - Thanks @gewarren!
- **Add release notes for 4.0.3** ([#7530](https://github.com/dotnet/machinelearning/pull/7530))
- **Update release-4.0.3.md** ([#7535](https://github.com/dotnet/machinelearning/pull/7535))
- **Add a doc with information about components and dependencies** ([#7537](https://github.com/dotnet/machinelearning/pull/7537))