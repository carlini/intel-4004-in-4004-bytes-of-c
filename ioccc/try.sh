#!/usr/bin/env bash
#
# try.sh - demonstrate IOCCC entry
#

# make sure CC is set so that when we do make CC="$CC" it isn't empty. Doing it
# this way allows us to have the user specify a different compiler in an easy
# way.
if [[ -z "$CC" ]]; then
    CC="cc"
fi

make CC="$CC" all

# clear screen after compilation so that only the entry is shown
clear

read -r -n 1 -p "Press any key to run: echo '5+3-oS' | ./prog redacted.bin: (please be very patient) "
echo 1>&2
echo "5+3-oS" | ./prog busicom_141pf.bin
echo 1>&2

echo "Now go forth and calculate!" 1>&2
