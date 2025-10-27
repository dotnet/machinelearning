# [ML.NET](http://dot.net/ml) 4.0.3

## **Bug Fixes**
- **[release/4.0] Improve unique directory generation for temp files** ([#7528](https://github.com/dotnet/machinelearning/pull/7528))
  - Compatibility note: This change resolves a performance problem where past versions of ML.NET would leave behind folders with the pattern `ml_dotnet\d+` in the temp directory, which would cause model opening performance to degrade.  This fixes the problem.  You may also wish to delete these empty folders once after updating.

    Using powershell:
    ```powershell
    Get-ChildItem "$env:TEMP" -Directory -Filter "ml_dotnet*" | Remove-Item -Recurse -Force
    ```

    Using bash
    ```bash
    find "$TEMP" -type d -name "ml_dotnet*" -exec rm -rf {} +
    ```
    


## **Build / Test updates**
- **[release/4.0] Update dependencies from dotnet/arcade** ([#7470](https://github.com/dotnet/machinelearning/pull/7470))
- **[release/4.0] Use arcade script for installing MacOS dependencies** ([#7534](https://github.com/dotnet/machinelearning/pull/7534))
