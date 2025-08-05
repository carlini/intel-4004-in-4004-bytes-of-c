# An Intel 4004 in 4004 bytes of C

This program is a feature-complete Intel 4004 emulator that's capable of
emulating the original Busicom 141-pf calculator ROM---the application
for which the 4004 was originally designed.

To make things exciting, this program doesn't emulate the 4004 the "normal" way.
Instead, this program is actually just a logic gate simulator, with the entire 4004
encoded as a circuit that's then emulated.
And because we're 50 years in the future, this gate-level emulated calculator
runs at a few hundred instructions per second which
is actually fast enough to be usable---just as long as you're not in a hurry.

This code was written for the 28th IOCCC in 2025, and so it contains two directories.
The directory `ioccc` has the code necessary to reproduce my IOCCC entry,
and the directory `4004bytes` has esentially the same program, but with a few
tweaks so that the program fits in 4004 bytes instead of 4993.


## Documentation

I've written up some [extensive documentation about this program on my website](https://nicholas.carlini.com/writing/2025/ioccc-intel-4004-in-4004-bytes-c.html),
if you'd like to read ~10,000 words explaining how this ~4000 byte program works.


## How to run

This project contains two directories, which are nearly identical. The `ioccc/` directory has the code to reproduce my IOCCC entry (exactly as it was when I submitted it, sorry for not cleaning anything up really...) and the `4004bytes/` directory has the tweaks necessary to turn the 4993-byte (but 2503-non-whitespace-byte) program into a 4004-byte program. Within each directory you can run

```
bash try.sh
```

to see a demo of the program run.


### Example calculations

You might want to try a few of these example calculations. On my computer these
take somewhere between 3 and 10 minutes to complete.

- Calculate 2+3: `2+3+=` (should return 5)
- Calculate 85*72: `85*72=` (should return 6120)
- Calculate 85*72*4: `85*72*o4=" (should return 24480)
- Calculate 85/72: `85/72=` (should return 1.1805555555555)
- Use the memory functions `5M4Mr` (should return 9)
- Memory with multiply `12*5=M3Mr` (should return 63)
- Calculate square roots: `2S` (should return 1.4142135623730)
- Use the sub-totals: `1+2+o5+o=` (should return 8)

Longer calculations can be performed but should be done "by hand" waiting
appropriately because the Busicom ROM only can hold only a limited number of
buffered keystrokes at a time. Care must be taken to not get unlucky when entering
keystrokes manually or else the wrong calculation will be performed. (This is
not a bug in the above C program, but a bug in the original ROM, which had
subtle timing race conditions that in practice were not real problems because
the calculator ran several thousand times faster than this emulated version
does.)


## License

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed in the hope that it will be useful, but 
WITHOUT ANY WARRANTY; without even the implied warranty of 
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
General Public License for more details.

You should have received a copy of the GNU General Public License 
along with this program. If not, see <http://www.gnu.org/licenses/>.