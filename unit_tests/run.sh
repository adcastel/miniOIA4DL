#!/bin/bash
for test in $(ls *.py)
do
    PYTHONPATH=. python3 $test
done
