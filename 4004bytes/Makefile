all:
	python3.9 verilog.py quiet
	rustc -O -o encoder encoder.rs
	./encoder
	gcc -o prog -O3 prog.c
	python3 assembler.py -o run.4004 test.asm
