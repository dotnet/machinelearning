$LIBS_LOCATION="..\..\..\artifacts\bin\Native\x64.Debug"
#ls $LIBS_LOCATION
$VENV_NAME=".venv"
# $PYTHON_NAME="python3"
$PYTHON_NAME="python"
Write-Host ("Should be installing in directory [" + $VENV_NAME + "]")
& $PYTHON_NAME -m venv $VENV_NAME
& ($VENV_NAME + "/Scripts/Activate.ps1")
& $PYTHON_NAME -m pip --proxy http://proxy-chain.intel.com:911 install -r requirements.txt
#$env:PATH=("..\..\artifacts\bin\Native\x64.Debug;" + $env:PATH)
$env:PATH=("$LIBS_LOCATION;" + $env:PATH)
& $PYTHON_NAME run_bench.py
deactivate

