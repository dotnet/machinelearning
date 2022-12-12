$VENV_NAME=".venv"
# $PYTHON_NAME="python3"
$PYTHON_NAME="python"
Write-Host ("Should be installing in directory [" + $VENV_NAME + "]")
& $PYTHON_NAME -m venv $VENV_NAME
& ($VENV_NAME + "/bin/Activate.ps1")
& $PYTHON_NAME -m pip install -r requirements.txt
& $PYTHON_NAME run_bench.py
