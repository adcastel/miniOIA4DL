#!/bin/bash
folder="unit_tests/"
utests=$(ls *.py)
cd ..
for test in $utests
do
    PYTHONPATH=. python3 ${folder}${test}
done
