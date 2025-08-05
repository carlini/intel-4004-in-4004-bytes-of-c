// Copyright (C) 2025, Nicholas Carlini <nicholas@carlini.com>.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#include<stdlib.h>
#include<stdio.h>
#include<fcntl.h>

#define W for(i=0;i<tmp;i++) 
#define INC_Q *Q++

#include "lz.h"

#define BIG 1<<24

int i;
int prog_src[BIG];
int insns[BIG];
int buf[BIG];
int functions[9];
int* z;
int tmp;
int tmp2;

int arguments[BIG];

int fraction;
int radix = 123;

int TT;
int print_uid;


int byte_offset;

int* depends[BIG];
int depends_ctr[BIG];
int reset_key_matrix;
int read_or_reset;
int pycounter;
int* charset;

int offset;
int color;

char v2[BIG], *trap_id;

long range = 1;


/*
  This function converts the compressed circuit into an uncompressed format.
  The input is a sequence of opcodes and arguments:
  - OP 0  |  A: WIRE that directly connects to prior gate A
  - OP 1  |  A: NOT the value of A
  - OP 2  |  A, B, C: OR the values of A and B
  - OP 3  |  A, B, C: AND the values of A and B
  - OP 4  |  A, B, C: XOR the values of A and B
  - OP 5  |  A, B, C: MULTIPLEXOR that computes the ternary a ? b : c
  - OP 6  |  REP_COUNT, NUM_ARGS, [offset, incr]* 
  - OP 7  |  FUNC_ID, NUM_ARGS, [argument]*
*/
int* cluster(int* Q) {
  int top;

  int *prev_start=Q, *now;

  int num_change;

  while ((top=INC_Q) < 9) {
    int a=(z-insns)/4, b=0;
    
    now = Q-1;

    // Ops less than 6 are direct expressions, not REPEAT or FUNCALL
    if (top < 6) {
      *z = top;

      // Compute how many more arguments we have to load for the ops
      //     REF: 1
      //     NOT: 1
      //     OR:  2
      //     AND: 2
      //     XOR: 2
      //     MUX: 3
      tmp = 1+(top>1)+(top>4);
      W {
        tmp2=INC_Q;
        // We need to convert from relative pointers to a fixed actual address
        // For arguments this has been done already, but otherwise we have to
        // re-calculate the offset. We're writing to the instruction at (z-insns)/4
        // so just subtract from that to recover.
        // the assignment here is to cast to an int, but in less space
        top = z[i+1] = tmp2 < 2e5 ? a - tmp2 : arguments[tmp2=tmp2-2e5];

        // In order to make things run ast, we need to know what gates depend on other gates
        // So store this here.
        if (top)
          depends[top][depends_ctr[top]++] = a;
      }
      z += 4;
    } else if (top ^ 7) { // function call
      // A function call goes and runs a small number of instructions and returns back.
      // Nested functions aren't possible.
      // So to call a function, we just set the global arguments buffer wih the correct arguments,
      // write out the data of the function, and then call cluster() on that location.
      
      tmp2 = INC_Q;
      tmp = INC_Q;
      W {
        arguments[i] = a += INC_Q;
      }

      // Copy to buf the current compressed ops
      // we'd like to just do a reference,
      // but if the called function uses loops then they modify the code in-place
      // so we have to copy before the function call
      tmp=BIG;
      W
        buf[i] = prog_src[i];
      cluster(buf+functions[tmp2]);
    } else { // repeat
      // A repeat is a loop that runs the prior instruction a variable number of times,
      // and also allows for updating the arguments to the prior instruction by a
      // constant scalar on each step.

      // To implement this, we use some self-modifying code:
      // 1. Decrement the argument that says how many repeats to do
      // 2. Adjust the other arguments from the prior instruction as necessary
      // 3. Go back 2 instructions if the counter is greater than zero.
      //    (on the next iteration of the loop we'll modify the arguments again).
      a=0;
      int number_of_times_to_duplicate = --INC_Q; // MAKETMP2
      num_change = tmp = INC_Q;
      if (number_of_times_to_duplicate) {
        W {
          b+=Q[num_change];
          prev_start[a+=INC_Q] += b;
        }
        Q = prev_start;
      } else {
        Q += num_change * 2;
      }
    }
    prev_start = now;
  }
  return Q;
}

/*
  Arithmetic decoding here. This uses a base-1968 encoding to specially take
  advantage of the IOCCC submission rules that whitespace characters following
  any braces don't count against the character limit. This function is almost
  verbatim lifted from Bellard's prior IOCCC submission.
*/
int get_bit(int ctx) {
  if (range < radix) {
    range *= radix;
    fraction *= radix;

    tmp = INC_Q;
    fraction += (tmp - 1 - ( tmp > 10 ) - ( tmp > 13 ) - ( tmp > 34 ) - ( tmp > 92 ));
  }
  int *counts = insns + ctx * 2;
  int split = range * -~*counts / (*counts + counts[ 1 ] + 2); // MAKETMP
  int the_bit = fraction >= split; // MAKETMP2
  fraction -= split*the_bit;
  range = the_bit ? range-split : split;

  counts[the_bit]++;
  return the_bit;
}


/*
  Read an integer out of a bitstream.
*/
int get_integer(int tmp, int ctx) {
  int subtract_it = 1<<tmp;
  int result_ans = 1;
  ctx*=99;
  while (!get_bit(++tmp+ctx));
  tmp--;
  W {
    result_ans = result_ans*2 | get_bit(ctx);
  }
  return result_ans - subtract_it;
}

int OFF1 = 4; // MAKEINLINE
int OFF2 = 1; // MAKEINLINE
int LITSIZE = 3; // MAKEINLINE


/*
  Used twice, to load varios bits of binary into the circuit:
  1. The program ROM, read from user input
  2. The micro-op table, as part of lz.h
 */
void fix_const_help(int depth, int writesize) {
  for (fraction = 0; fraction < depth; fraction++) {
    tmp = 1<<writesize;
    W {
      insns[byte_offset++*4+1] =  buf[i]>>fraction&1;
    }
  }
}

/*
  A single main function that does a bunch of things in sequence
  to save space. The overall flow of the program is to first
  decode the lz.h and LZ-decompress it, then turn that compressed
  circuit representation into actual logic gates, and then finally
  run that circuit.
 */
int main(int argc, char** argv) {

  if (argc<2) exit(42);
  fcntl(0,4,O_NONBLOCK);

  /////////////////////////////////
  /* INLINED FUNCTION DECOMPRESS */
  /////////////////////////////////
  
  int* Q = prog_src;
  int tmp,i,j = get_integer(9, 0);

  while (j--) {
    if (get_bit(1)) {
      z = Q - get_integer(OFF1, 2) - 1;
      tmp = get_integer(OFF2, 3) + 1;
      W {
        INC_Q = *z++;
      }
    } else {
      INC_Q = (1-2*get_bit(8)) * get_integer(LITSIZE, 9);
    }
  }
  
  Q = prog_src;
  tmp = TT = INC_Q;
  trap_id = v2+INC_Q;
  print_uid = INC_Q;
  W {
    depends[i] = malloc(7e4);
  }

  tmp=INC_Q; // hard coded to 5 functions
  W {
    functions[i] = INC_Q;
  }
  int max_args = INC_Q; // MAKETMP2
  z = insns;
  
  
  Q = cluster(prog_src+max_args);

  ////////////////////////////////
  /* INLINED FUNCTION FIX_CONST */
  ////////////////////////////////

  // where should we be writing to?
  byte_offset = INC_Q;

  FILE *file = fopen(argv[1], "r");
                     
  tmp=4096; // read 2^12 bytes of program ROM, TODO could shrink here
  W {
    buf[i] = fgetc(file);
  }

  fix_const_help(8, 12);

  // Now we fill the uops, starting with the offset
  byte_offset = INC_Q;

  // uops are encoded as 6-bit ints for better compression; unpack them
  // we store a maximum of 16 uops per instruction
  tmp=16;
  W {
    // there are a total of 256 instructions
    j=256;
    while (j--) {
      buf[j] = Q[j*16+i];
    }
    // offset the writing appropriately
    fix_const_help(6, 8);
  }

  
  charset = Q+4096;


  ////////////////////////////////
  /* INLINED FUNCTION SIMULATE  */
  ////////////////////////////////
  /*
    Actually run the reconstructed gates, and implement the read/write operations.
    At this point in time everything has been decompressed and we just need to loop over
    all of the gates and execute them.
    
    To make things go fast we only run the gates that need an update, which allos us to
    skip over the parts of the circuit that remain constant and are only updated
    occasionally, like the memory reads.
  */

  // Store which gates need an update
  unsigned long *needs_update = calloc(TT,16),
    tmpll;


  while (1) {
    int command;

    int ctr = 0;
    int todo,*Q;
    int prev_val;

    while (ctr < TT) {

      tmp=3;
      while (tmp--) {
        int divide = 1<<6*tmp; // MAKETMP2
        todo = ctr/divide;

        tmpll = needs_update[offset + TT/64*tmp + todo/64] >> todo%64;

        // To make things fast we have a 3-level tree for which instructions need to be run.
        // Each level is a 64 bit integer with a 1 meaning we should recurse down a layer,
        // with the final layer meaning we should handle the instruction.

        while (tmpll&1) {
          ctr = ctr - ctr%divide + divide;
          tmpll/=2;
          tmp = 0; // break out of the above loop
        }
      }

      Q=insns+ctr*4;
      command = INC_Q;

      // Load the potential arguments for this instruction.
      i = v2[INC_Q];
      tmp = v2[INC_Q];
      tmp2 = v2[INC_Q];

      // We have six possible commands REF, NOT, OR, AND, XOR, MUX
      // Here we're going to evaluate with a series of ternary ops.

      prev_val = v2[ctr];
      v2[ctr] = command<2 ?
        i^command // either 0 or 1
        :
        // either 2,3,4,5
        command<4 ?
          command^3 ?
            i|tmp :
            i&tmp
          :
          command^5 ?
            i^tmp :
            i?tmp:tmp2;

      // If the value has changed then we need to execute anything that depends on this one
      // because those are now stale too.
      if (prev_val ^ v2[ctr]) {
        
        tmp2 = depends_ctr[ctr];
        while (tmp2--) {
          todo = depends[ctr][tmp2];
          
          tmp=3;
          W {
            needs_update[(offset ^ TT * (todo <= ctr)) + TT/64*i + (todo>>6*-~i)] &= ~(1L<<(todo>>6*i)%64);
          }
          
        }
      }


      ctr++;
    }
    
    // Swap the two arrays

    tmp=TT/16;
    W {
      needs_update[offset + i] = -1L;
    }

    offset ^= TT;


    // Handle the I/O of the calculator
    // If the TRAP gate is set then we might need to read or write something.
    {
      char *Q = trap_id;
    
      if (INC_Q) {
        color |= INC_Q;

        int is_write = INC_Q;  // MAKETMP
        int rom_or_ram = !INC_Q; // MAKETMP2


        pycounter += rom_or_ram & !is_write;
        if (is_write & rom_or_ram & INC_Q) {
          tmp = 18;
          putchar(32+color*13);
          W {
            int idx = ((i+3)%18-(i>15)); // MAKEINLINE
            putchar(charset[i*18+*(int*)(v2+print_uid+idx*21)%18]);
          }
          color = 0;
          puts("");
        }
      }

      // Every 128 JCN instructions we should send the next keypress.
      // 128 is hard-coded in the circuit. Don't change.
      if (pycounter - reset_key_matrix > 128) {
        reset_key_matrix = pycounter;
        read_or_reset = !read_or_reset;
        tmp2 = read_or_reset ? 0 : charset[325 + getchar()];
        tmp=8;
        W {
          INC_Q = tmp2>>i&1;
        }
      }
    }
  }
}
