#!/usr/bin/env bash




EXECUTION_DIR=$(dirname "$0")




# ========================= BEGIN Test Execution =============================

echo ----- start $(date) ===============  To repro directly: =====================================================

echo pushd $EXECUTION_DIR

echo ${runCommand}

echo popd

echo ===========================================================================================================

pushd $EXECUTION_DIR

${runCommand}

test_exitcode=$?

if [[ -s testResults.xml ]]; then

  has_test_results=1;

fi;

popd

echo ----- end $(date) ----- exit code $test_exitcode ----------------------------------------------------------

# ========================= END Test Execution ===============================




# The tests either failed or crashed, copy output files

if [[ "$test_exitcode" != "0" && "$HELIX_WORKITEM_UPLOAD_ROOT" != "" ]]; then

  tar -czf $HELIX_WORKITEM_UPLOAD_ROOT/TestOutput.tar.gz $EXECUTION_DIR/TestOutput/

fi




# The helix work item should not exit with non-zero if tests ran and produced results

# The xunit console runner returns 1 when tests fail

if [[ "$test_exitcode" == "1" && "$has_test_results" == "1" ]]; then

  if [ -n "$HELIX_WORKITEM_PAYLOAD" ]; then

    exit 0

  fi

fi




exit $test_exitcode