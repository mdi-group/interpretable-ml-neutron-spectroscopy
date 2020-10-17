#!/bin/bash

find "Horace" -size 0 -exec rm -f '{}' \;
find "Horace" -name _docify -exec rm -rf '{}' \;

find "Herbert" -size 0 -exec rm -f '{}' \;
find "Herbert" -name _docify -exec rm -rf '{}' \;
rm -rf "Herbert/herbert_core/applications/docify"

cp -f horace_sqw_red.m brillem/+brillem/@Brille/

mcc -m -a Horace/horace_core -a Herbert/herbert_core -a spinw -a brillem -a dimer_model.m -a pcsmo_model.m -d . -v -N model_eval.m
