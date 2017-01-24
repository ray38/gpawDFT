#!/bin/sh
nh=doc/platforms/Linux/Niflheim
rm -rf build
cmd="cd $PWD && python setup.py --remove-default-flags build_ext"
ssh slid "$cmd --customize=$nh/x3455.py > x3455.log 2>&1"
ssh surt "$cmd --customize=$nh/sl230s.py > sl230s.log 2>&1"
ssh muspel "$cmd --customize=$nh/dl160g6.py > dl160g6.log 2>&1"
ssh sylg "$cmd --customize=$nh/el7.py > el7.log 2>&1"
