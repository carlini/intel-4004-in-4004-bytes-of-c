# Can we always take only the lower 0xF of b?
## Try to figure out which write wire to use is most efficient

# might be able to save some space by always clearining high 8 bits
# on the move from b -> c

# TODO manually un-expand some [5] REPs to allow LZ77 to work better
# that is, instead of compressing all [a,a,a,a,a] + c -> 5x[a] + c
# it might be better to do 4x[a] + [a] + c for compression if [a] + c appears elsewhere
# (this happens especialyl in the ROM section)

# TODO automatically increment PC when we hit the end of the op?

# TODO better to use additional registers + a pointer as the PC stack?


import pdb
import verilog
import gzip
from verilog import Signal, Bits, const, sim, circ, CIRC_PATTERNS
import time
import numpy as np

REG_DEPTH = 8

def mux(cond, a, b):
    return cond.mux(a,b)#(cond & a) | (~cond & b)

def is_eq(cond, i):
    o = cond.value[0].eq(const(i&1))
    for c in cond.value[1:]:
        i >>= 1
        z = c.eq(const(i&1))
        o = o & z
        #print("Test", c)
    return o            

def dff_half(inp, clock, n=1):
    if isinstance(inp, Bits):
        q = Bits([Signal() for _ in range(n)])
    else:
        q = Signal()
    out = mux(clock, inp, q)
    q.connect(out)

    return q

def dff(inp, clock, n=1):
    q = dff_half(inp, ~clock, n)
    q = dff_half(q, clock, n)
    return q

@circ(5, 1)
def dff_circ(inp, clock):
    q = dff_half(inp, ~clock, 5)
    q = dff_half(q, clock, 5)
    return q

def clock(latency):
    out = Signal(how='BAD')
    inv = ~out
    delays = [Signal(how='BAD') for _ in range(latency)][::-1]
    for a,b in zip(delays,delays[1:]):
        b.connect(a)
    delays[0].connect(inv)
    out.connect(delays[-1])
    return out

#def inc(bits, send_carry=False):
#    carry = bits.value[0] | ~bits.value[0]  #const(1)
#    out = []
#    for b in bits:
#        #out.append(mux(b, ~carry, carry))
#        out.append(b ^ carry)
#        carry = b & carry
#    if send_carry:
#        return Bits(out + [carry])
#    return Bits(out)

def inc(bits, send_carry=False):

    masks = [const(1)]
    for b in bits:
        masks.append(masks[-1] & b)

    out = []
    for b,x in zip(bits,masks):
        out.append(b^x)
    carry = masks[-1]
    if send_carry:
        return Bits(out + [carry])
    return Bits(out)

@circ(1, 1, 1)
def full_add(in1, in2, in3):
    low1 = in1^in2
    low2 = low1^in3
    high = (in1 & in2) | (low1 & in3)
    return Bits([low2, high])

def add(in1, in2, carry, return_carry=False):
    out = []
    carry = carry | carry
    for a,b in zip(in1.value, in2.value):
        c, carry = full_add(a, b, carry).value
        out.append(c)
    if return_carry:
        return Bits(out), carry
    return Bits(out)


def any_set(bits):
    r = bits.value[0]
    for x in bits.value[1:]:
        r = r | x
    return r

def all_set(bits):
    r = bits.value[0]
    for x in bits.value[1:]:
        r = r | x
    return r

#@circ(1, 1, 1, 12, 12)
#def one_reg(clock, enable_write, do_write, data, output1):
#    sig = Bits([Signal() for _ in range(12)])
#    out = dff(sig, clock, 12)
#    sig.connect(mux(enable_write & do_write, data, out))
#    return Bits(out.value +  (output1 | (out & do_write)).value)

@circ(1, 1, 1, 12, 12)
def one_reg(clock, enable_write, do_write, data, output1):
    ok = enable_write & do_write
    out = dff(data, clock & ok, 12)
#    return Bits(out.value +  (output1 | (out & do_write)).value)
    return Bits(out.value +  mux(do_write, out, output1).value)

@circ(REG_DEPTH, REG_DEPTH)
def eq_by_inc(counter, reg_write):
    #return Bits(inc(counter).value + [~any_set(mux(counter, ~reg_write, reg_write))])
    return Bits(inc(counter).value + [~any_set(counter^reg_write)])

def is_right(reg_write, offset=16):
    out = []
    counter = Bits([const(0) for _ in range(len(reg_write.value))])

    # TODO DOUBLE CHECK WHY OFFSETS CHANGE twelve nops for compression
    xs = [const(0) for _ in range(offset)] 
    #[x.connect(x) for x in xs]
                    
    
    for i in range(1<<REG_DEPTH):
        o = eq_by_inc(counter, reg_write)
        counter = Bits(o.value[:REG_DEPTH])
        out.append(o.value[REG_DEPTH])

        #o.value[5].track("o"+str(i))

    return Bits(out)

def muxn(cond, choices, early=True):
    #print("Called", len(choices), force)
        
    if 2**len(cond.value) != len(choices):
        raise
        return muxn(cond, list(choices) + [const(0)] * (2**len(cond.value) - len(choices)))

    if len(choices) == 1: return choices[0]

    if early and all(x is choices[0] for x in choices):
        return choices[0]
    
    if early and all(choices[-1] is x for x in choices[len(choices)//2:]):
        return mux(cond.value[-1],
                   choices[-1],
                   muxn(Bits(cond.value[:-1]), choices[:len(choices)//2]))

#    return mux(cond.value[-1],
#               muxn(Bits(cond.value[:-1]), choices[len(choices)//2:]),
#               muxn(Bits(cond.value[:-1]), choices[:len(choices)//2]))
    
    return muxn(Bits(cond.value[1:]), [mux(cond.value[0], b, a) for a,b in zip(choices[::2], choices[1::2])])

#z = muxn(Bits([const(0), const(1)]), [const(0), const(1), const(0), const(1)])
#sim(100, z)
#print(z.value)
#exit(0)

@circ(4, *[1]*16)
def do_mux16(a, *regs):
    assert len(a.value) == 4
    assert len(regs) == 16
    return muxn(a, regs, early=False)

#@circ(3, *[1]*8)
#def do_mux8(z, *regs):
#    return muxn(z, regs)

def regfile(clock, reg_write, data_1, enable_reg_write):
    mem = []

    reg_read = reg_write
    #reg_write_1 = [is_eq(reg_write_1, i) for i in range(32)]
    #reg_write_2 = [is_eq(reg_write_2, i) for i in range(32)]
    #print("RUN", reg_write_1.value)
    reg_write = is_right(reg_write).value
    #reg_write_2 = is_right(reg_write_2).value

    output1 = Bits([const(0) for _ in range(12)])


    # xxx0 0000 <- for main data
    # xxx1 0000 <- for control data
    
    # 1110 0000 <- for registers
    # 1111 0000 <- for for special values

    
    for i in range(256):
        #if i == 25:
        #    enable_reg_write.alert = lambda: print("ENABLE WRITE 99", enable_reg_write.value)
        #    reg_write[i].alert = lambda: print("DO WRITE 99", enable_reg_write.value)
        #    for z in range(4):
        #        data_1.value[z].alert = lambda: print("Changed data", data_1)
        out = one_reg(clock, enable_reg_write, reg_write[i], data_1, output1)
        mem.append(Bits(out.value[:12]))
        output1 = Bits(out.value[12:])

    
    #output1 = Bits([do_mux32(reg_read, *[x.value[i] for x in mem]) for i in range(12)])
    #output2 = Bits([do_muxn(reg_read_2, *[x.value[i] for x in mem]) for i in range(12)])
    #output2 = [const(0)]*12


    return output1, mem

def mux8(cond, a,b,c,d,e,f,g,h):
    #return muxn(cond, [a,b,c,d,e,f,g,h])
    return mux(cond.value[0],
               mux(cond.value[1],
                   mux(cond.value[2],
                       h,d),
                   mux(cond.value[2],
                       f,b)),
               mux(cond.value[1],
                   mux(cond.value[2],
                       g,c),
                   mux(cond.value[2],
                       e,a)))

## def rom1(addr, inc_addr, arr, depth, force=False, logger=[]):
##     low_byte = Bits([muxn(addr, [const((x>>i)&1) for x in arr]) for i in range(depth)])
##     return low_byte
##     #if inc_addr is None:
##     #    return Bits([const(0)]*depth)
##     #return Bits([const(0)]*depth), Bits([const(0)]*depth)
##     
##     def doit(addr):
##         
##         out = []
##         #print("Depth", len(addr.value), depth)
##         for i in range(depth):
##             addr = addr.clone()
##             zar = [const((x>>i)&1) for x in arr]
##             out.append(muxn(addr, zar, force=force))
##         return Bits(out)
## 
##     
##     logger.append(len(addr.value[0].signals))
##     #print('a')
##     low_byte = doit(addr)
##     if inc_addr is None:
##         return low_byte
##     logger.append(len(addr.value[0].signals))
##     #print('b')
##     next_byte = doit(inc_addr)
##     return low_byte, next_byte

def rom(addr, inc_addr, arr, depth, force=False, logger=[]):

    if len(arr) < 2**len(addr.value):
        arr = arr + [0] * (2**len(addr.value) - len(arr))

    def my_muxn(cond, choices):
        #assert all(2**len(cond.value) == len(x) for x in choices)
        #print('call', len(Signal.signals))
    
        if len(choices[0]) == 1:
            return [x[0] for x in choices]
    
        return my_muxn(Bits(cond.value[1:]), [[mux(cond.value[0], b, a) for a,b in zip(x[::2], x[1::2])] for x in choices])

    # the problem is that const(1) is not just one token

    def myconst(x):
        if force:
            return Signal(how=('VAR', 0),
                          fn=lambda: 0)
            
        if x == 0:
            return Signal(how=('VAR', -2),
                          fn=lambda: 0)
        else:
            return Signal(how=('VAR', -1),
                          fn=lambda: 1)

    
    def doit(addr):
        if False:
            #print("GOGO")
            logger.append((len(addr.value[0].signals), arr))
            sig = iter([myconst(0) for _ in range(depth * len(arr))])
            return Bits(my_muxn(addr, [[next(sig) for x in arr] for i in range(depth)]))
        else:
            logger.append((len(addr.value[0].signals), arr))
            #sig = iter([const((x>>i)&1) for x in range(len(arr)) for i in range(depth)])
            
            #return Bits(my_muxn(addr, [[next(sig) for x in arr] for i in range(depth)], force=force))
            return Bits(my_muxn(addr, [[myconst((x>>i)&1) for x in arr] for i in range(depth)]))

    low_byte = doit(addr)

    return low_byte

ACC_REG = 0xF8
CARRY_REG = 0xF9
TMP_REG = 0xFA
RAM_REG = 0xFB
PC_REG = 0xFC

def assemble(insns):

    ## loadstore
    IMM, IMMBYTE, RAM_INDIRECT, RAM_STATUS, REG, REG2, REGP0, REGP1, ACC, CARRY, TMP, RAM_ADDR, PC, PC1, PC2, PC3 = range(16)


    _ZERO, _, _DUP, _NOT,   _ADD0, _ADD1, _ADDC, _ADDCI,   _GETCARRY, _, _KBP, _DAA,  _MUXTMP, _, _MUXNOT, _MUX  = range(16)
    _PCHIGH, _, _TESTREAD, _ROMREAD,   _JCN, _, _RAMWRITE, _ROMWRITE,   _LEFT4, _, _MAKEBYTE, _,  _INC, _DOUBLE, _, NEXT  = range(16,32)

    # LOAD  00 RRRR
    # STORE 01 RRRR
    # ALU   1 XXXXX

    def load(x):
        return (0<<4) | x
    def store(x):
        return (1<<4) | x
    def alu(x):
        return (2<<4) | x

    NEXT = 0x3F

    step = [load(PC),
            alu(_INC),
            store(PC),
            NEXT]
    

    def NOP():
        return [*step]
    
    def LDM(imm):
        return [load(IMM),
                store(ACC),
                *step]

    def LD(reg):
        return [load(REG),
                store(ACC),
                *step]

    def XCH(reg):
        return [load(REG),
                store(TMP),
                load(ACC),
                store(REG),
                load(TMP),
                store(ACC),
                *step
                ]

    def ADD(reg):
        return [load(REG),
                load(ACC),
                alu(_ADDC),
                store(ACC),
                alu(_GETCARRY),
                store(CARRY),
                *step]

    def RAL():
        return [load(ACC),
                load(ACC),
                alu(_ADDC),
                store(ACC),
                alu(_GETCARRY),
                store(CARRY),
                *step]

    def RAR():
        return [load(CARRY),
                alu(_LEFT4),                
                load(ACC),
                alu(_ADD0), # Cabcd
                store(CARRY), # Cabcd
                alu(_DOUBLE), # Cabcd0
                alu(_DOUBLE), # Cabcd00
                alu(_DOUBLE), # Cabc d000
                alu(_ZERO), # NOP
                alu(_GETCARRY), # Cabc
                store(ACC),
                
                *step]
    
    
    def SUB(reg):
        return [load(REG),
                alu(_NOT),
                load(ACC),
                alu(_ADDCI),
                store(ACC),
                alu(_GETCARRY),
                alu(_NOT),
                store(CARRY),
                *step]
    
    def INC(reg):
        return [load(REG),
                alu(_INC),
                store(REG),
                *step]


    def JUN(reg):
        return [load(PC),
                alu(_INC),
                store(PC),
                load(IMM),
                alu(_LEFT4),
                alu(_LEFT4),
                load(IMMBYTE),
                alu(_ADD0),
                store(PC),
                NEXT]

    def RDR():
        return [alu(_ROMREAD),
                store(ACC),
                *step]

    def WMP():
        return [alu(_RAMWRITE),
                *step]

    def WRR():
        return [alu(_ROMWRITE),
                *step]

    
    def JCN(reg):
        return [alu(_TESTREAD),
                load(PC),
                alu(_INC),
                store(PC),
                alu(_INC),
                load(IMMBYTE),
                alu(_JCN),
                alu(_MUXNOT),
                alu(_PCHIGH),
                store(PC),
                NEXT]
    
    def JIN(reg):
        return [load(REG2),
                alu(_LEFT4),
                load(REG),
                alu(_ADD0),
                alu(_PCHIGH),
                store(PC),
                NEXT]

    # TODO can I get rid of MUXTMP?
    def ISZ(reg):
        return [load(REG), # increment reg
                alu(_INC),
                store(REG),

                store(TMP), # tmp=1 if reg=0

                load(PC),
                alu(_INC),
                store(PC), # increment PC

                alu(_INC), # inc PC again
                load(IMMBYTE),
                load(IMM), # NOP
                alu(_MUXTMP),
                alu(_PCHIGH),
                alu(_DUP), # NOP
                alu(_DUP), # NOP
                store(PC)]
    
    def JMS(reg):
        return [load(PC2),
                store(PC3),
                load(PC1),
                store(PC2),
                load(PC),
                alu(_INC),
                store(PC),
                store(PC1),
                load(IMM),
                alu(_LEFT4),
                alu(_LEFT4),
                load(IMMBYTE),
                alu(_ADD0),
                alu(_DUP),
                store(PC)]

    def BBL(reg):
        return [load(IMM),
                store(ACC),
                load(PC1),
                alu(_INC),
                store(PC),
                load(PC2),
                store(PC1),
                load(PC3),
                store(PC2),
                ]
    
    def IAC():
        return [load(ACC),
                alu(_INC),
                store(ACC),
                alu(_GETCARRY),
                store(CARRY),
                *step]

    def DAC():
        return [alu(_ZERO),
                alu(_NOT),
                load(ACC),
                alu(_ADD0),
                store(ACC),
                alu(_NOT),
                alu(_ZERO), # NOP
                alu(_GETCARRY),
                store(CARRY),
                *step]
    
    def STC():
        return [alu(_ZERO),
                alu(_INC),
                store(CARRY),
                *step]

    def CLC():
        return [alu(ZERO),
                store(CARRY),
                *step]

    def CMC():
        return [load(CARRY),
                alu(_NOT),
                store(CARRY),
                *step]

    def CMA():
        return [load(ACC),
                alu(_NOT),
                store(ACC),
                *step]
    
    def TCC():
        return [load(CARRY),
                store(ACC),
                alu(_ZERO),
                store(CARRY),
                *step]
    
    def CLB():
        return [alu(_ZERO),
                store(ACC),
                store(CARRY),
                *step]
    
    def CLC():
        return [alu(_ZERO),
                store(CARRY),
                *step]

    def KBP():
        return [load(ACC),
                alu(_KBP),
                store(ACC),
                *step]
    
            # TODO: fix DAA so that it sets carry if the result generates a carry
            # probably do this by writing some new circuit
    def DAA():
        return [
            #load(IMM),
            #alu(_ADD0),
            #load(ACC),
            #alu(_ADD0),
            #store(TMP), # now stored x+6 to tmp
            #load(ACC),
            #alu(_GETCARRY), # load the carry from x+6
            #alu(_MUXNOT),
            #load(TMP),
            #load(CARRY),
            #alu(_MUXNOT),
            #store(ACC),
            
            #load(ACC),
            alu(_DAA),
            store(ACC),
            alu(_GETCARRY), # load the carry from x+6
            store(CARRY),
            
            *step]

    def TCS():
        return [load(IMM),
                alu(_ZERO),
                alu(_ADDC),
                store(ACC),
                alu(_ZERO),
                store(CARRY),
                *step]
    
    def FIM(reg):
        return [load(PC),
                alu(_INC),
                store(PC),
                load(IMMBYTE),
                store(REG2),
                alu(_GETCARRY),
                store(REG),
                *step]


    
    def FIN(reg):
        return [load(PC),
                store(TMP),
                load(REGP1),
                load(REGP0),
                alu(_MAKEBYTE),
                alu(_PCHIGH),
                store(PC),
                load(IMMBYTE), # NOP
                load(IMMBYTE),
                store(REG2),
                alu(_GETCARRY),
                store(REG),
                load(TMP),
                alu(_INC),
                store(PC)]
                
                

    def SRC(reg):
        return [load(REG2),
                alu(_LEFT4),
                load(REG),
                alu(_ADD0), # loaded the byte
                store(RAM_ADDR),
                *step]

    def WRM():
        return [load(ACC),
                store(RAM_INDIRECT),
                *step]

    def RDM():
        return [load(RAM_INDIRECT),
                store(ACC),
                *step]

    def ADM():
        return [load(RAM_INDIRECT),
                load(ACC),
                alu(_ADDC),
                store(ACC),
                alu(_GETCARRY),
                store(CARRY),
                *step]

    def SBM():
        return [load(RAM_INDIRECT),
                alu(_NOT),
                load(ACC),
                alu(_ADDCI),
                store(ACC),
                alu(_GETCARRY),
                alu(_NOT),
                store(CARRY),
                *step]

    def WR0():
        return [load(ACC),
                store(RAM_STATUS),
                *step]
    def WR1():
        return [load(ACC),
                store(RAM_STATUS),
                *step]
    def WR2():
        return [load(ACC),
                store(RAM_STATUS),
                *step]
    def WR3():
        return [load(ACC),
                store(RAM_STATUS),
                *step]

    def RD0():
        return [load(RAM_STATUS),
                store(ACC),
                *step]
    def RD1():
        return [load(RAM_STATUS),
                store(ACC),
                *step]
    def RD2():
        return [load(RAM_STATUS),
                store(ACC),
                *step]
    def RD3():
        return [load(RAM_STATUS),
                store(ACC),
                *step]
    

    opcodes = [
        (NOP, 	"00000000"),
        (JCN, 	"0001DDDD AAAAAAAA"),
        (FIM, 	"0010RRR0 DDDDDDDD"),
        (SRC, 	"0010RRR1"),
        (FIN, 	"0011RRR0"),
        (JIN, 	"0011RRR1"),
        (JUN, 	"0100DDDD AAAAAAAA"),
        (JMS, 	"0101DDDD AAAAAAAA"),
        (INC, 	"0110DDDD"),
        (ISZ, 	"0111DDDD AAAAAAAA"),
        (ADD, 	"1000DDDD"),
        (SUB, 	"1001DDDD"),
        (LD, 	"1010DDDD"),
        (XCH, 	"1011DDDD"),
        (BBL, 	"1100DDDD"),
        (LDM, 	"1101DDDD"),
        (WRM, 	"11100000"),
        (WMP, 	"11100001"),
        (WRR, 	"11100010"),
        (WR0, 	"11100100"),
        (WR1, 	"11100101"),
        (WR2, 	"11100110"),
        (WR3, 	"11100111"),
        (SBM, 	"11101000"),
        (RDM, 	"11101001"),
        (RDR, 	"11101010"),
        (ADM, 	"11101011"),
        (RD0, 	"11101100"),
        (RD1, 	"11101101"),
        (RD2, 	"11101110"),
        (RD3, 	"11101111"),
        (CLB, 	"11110000"),
        (CLC, 	"11110001"),
        (IAC, 	"11110010"),
        (CMC, 	"11110011"), 
        (CMA, 	"11110100"),
        (RAL, 	"11110101"),
        (RAR, 	"11110110"),
        (TCC, 	"11110111"),
        (DAC, 	"11111000"),
        (TCS, 	"11111001"),
        (STC, 	"11111010"),
        (DAA, 	"11111011"),
        (KBP, 	"11111100"),
#        (DCL, 	"11111101"),
    ]

    table = [[] for _ in range(256)]

    for fn, bits in opcodes:
        if 'RR' in bits:
            for i in range(8):
                table[int(bits.split()[0].replace("RRR", "000"),2) + (i<<1)] = fn(i)
        elif 'DDDD' in bits:
            for i in range(16):
                table[int(bits.split()[0].replace("DDDD", "0000"),2) + i] = fn(i)
        else:
            table[int(bits,2)] = fn()

    for k,v in enumerate(table):
        y = 0
        #while len(v) < 15:
        #    v = [0] + v
        #assert len(v) == 15, "More than 16 uops for %02x"%k

        if len(v) == 16:
            assert v[-1] == 0x3F
            v.pop()
        
        assert len(v) < 16

        #for i,x in enumerate(v):
        #    if x == store(PC):
        #        assert i%8 == 6, "err 0x%02x"%k
            
        while len(v):
            y <<= 6
            y |= v.pop()
        table[k] = y


    labels = {}
    insn_stream = []

    offset = 1
    for x in insns.split("\n"):
        if len(x.split()) == 0: continue
        x = x.split(";")[0]
        if len(x.split()) == 0: continue

        if ':' in x:
            l, x = x.split(":")
            labels[l.strip()] = offset

        x = x.strip()

        if len(x) == 0: continue

        ops = dict(opcodes)[eval(x.split(" ")[0])]

        offset += len(ops.split())

            
    for x in insns.split("\n"):
        if len(x.split()) == 0: continue
        x = x.split(";")[0]
        if len(x.split()) == 0: continue
        x = x.split(":")[-1]
        x = x.strip()
        if len(x.split()) == 0: continue
        x = x.replace(",", " ")
        for _ in range(10):
            x = x.replace("  ", " ")

        for i in range(0,16,2):
            x = x.replace("R%xR%x"%(i,i+1), str(i//2))

        def fix(arg):
            if arg in labels:
                return labels[arg]
            if arg[0] == 'R':
                arg = arg[1:]
            remap = {'NC': 10,
                     'Z': 4}
            if arg in remap: return remap[arg]
            return int(arg)

        fn = eval(x.split(" ")[0])
        args = list(map(fix,x.split(" ")[1:]))

        #print(fn)
        try:
            opcode = [x[1] for x in opcodes if x[0] == fn][0]
        except:
            print("Uknown opcode", fn)
            exit(0)


        next_byte = None
        if len(opcode.split()) > 1:
            next_byte = args[-1]&0xFF
            args[-1] = args[-1] >> 8
            opcode = opcode.split()[0]

        if 'RRR' in opcode:
            assert args[0] < 8
            insn_stream.append(int(opcode.split()[0].replace("RRR", "000"),2) + (args[0]<<1))
        elif 'DDDD' in opcode:
            insn_stream.append(int(opcode.split()[0].replace("DDDD", "0000"),2) + args[0])
        else:
            assert len(args) == 0
            insn_stream.append(int(opcode.split()[0],2))

        if next_byte is not None:
            insn_stream.append(next_byte)


    return None, table, [0] + insn_stream + [0]*4

CLOCK_FREQ = 8

def make_4004(insns, fake=False, logger=[]):

    _zero = Signal()
    _zero.connect(_zero)
    _one = ~_zero
    
    
    _, table, load_insn_stream = assemble(insns)
    #print(" ".join(["%02x"%x for x in load_insn_stream]))

    #clock_inner = clock(32)
    clock_inner = clock(CLOCK_FREQ)
    
    addr = Bits([Signal(add_signal=False) for _ in range(12)])

    (clock_inner).track("clock inner")

    abort_early = Signal()
    
    addr_sub = Bits([Signal() for _ in range(4)])
    inc_addr_sub = inc(addr_sub, True)
    inc_addr_sub = inc_addr_sub

    clock_outer = inc_addr_sub.value[-1]

    #abort_early | abort_early
    
    addr_sub.connect(dff(inc_addr_sub | abort_early, clock_inner, 4))
    dff(inc_addr_sub, clock_inner, 4)  # TODO: remove this one
    (addr_sub).track("addr_sub")

    which_reg = Bits([Signal() for _ in range(REG_DEPTH)])
    do_reg_write = Signal()
    reg_write_data = Bits([Signal() for _ in range(12)])

    reg_output, regs = regfile(clock_inner, which_reg, reg_write_data, do_reg_write)

    insn_stream = Bits([Signal() for _ in range(8)])
    u_ops_base = Bits([Signal(add_signal=False) for _ in range(96)])

    u_op = Bits([Signal() for _ in range(6)])

    abort_early.connect(u_op.value[0] & u_op.value[1] & u_op.value[2] & u_op.value[3] & u_op.value[4] & u_op.value[5])
    
    reg_output.track("reg_output")
    acc = regs[ACC_REG]
    carry = regs[CARRY_REG].value[0]
    ram_indirect = regs[RAM_REG]
    addr.connect(regs[PC_REG], reuse=True)
    (regs[PC_REG]).track("addr")

    #(regs[25]).track("0x99")
    #regs[25].value[3].alert = lambda: print("CHANGED AND NOW ", regs[0x99])
    
    
    #exit(0)
    
    u_ops = [Bits(u_ops_base.value[i:i+6]) for i in range(0,96,6)]

    #u_op_ = muxn(addr_sub, u_ops)
    u_op_ = Bits([do_mux16(Bits(addr_sub.value), *[x.value[i] for x in u_ops]) for i in range(6)])
    #print('uu',u_op_.uids())
    u_op.connect(u_op_)
    (u_op_).track("u_op")

    do_alu = u_op.value[5]

    notu5 = ~u_op.value[5]
    do_load = ~u_op.value[4] & notu5
    do_store = u_op.value[4] & notu5

    
    (do_alu|do_alu).track("do_alu")
    do_load.track("do_load")
    do_store.track("do_store")

    # This stores the uop saying the index of the loadstore op we're going to make
    which_loadstore = Bits(u_op.value[:-2])
    (which_loadstore|which_loadstore).track("which loadstore")

    do_reg_write.connect(do_store)
    
    alu_op = Bits(u_op.value[:-1])

    insn_stream_before_ = Bits([Signal() for _ in range(8)])

    
    _low = mux(which_loadstore.value[0],
               Bits(insn_stream_before_.value[:8]),
               Bits(Bits(insn_stream.value[:4]).clone().value + [const(0) for _ in range(8)]))
    _high = Bits([const(0) for _ in range(8)])
    immediate_loaded_value = Bits(_low.value + _high.value)
    assert np.all(np.diff(immediate_loaded_value.uids())==1)
    
    # The two immediate loads are placed in 0000 and 0001 so as long as any of the high bits
    # are 1 then we want to be reading out of the "register file'.
    # 1. for standard register accesses to the 16 general purpose registers
    # 2. to my special 8 registers for the ACC, CARRY, TMP, RAM_ADDR, PC0/1/2/3
    # 3. for the RAM we have on the chip
    loaded_value = mux((which_loadstore.value[1] | which_loadstore.value[2] | which_loadstore.value[3]),
                       reg_output,
                       immediate_loaded_value)


    # Our loadstore value is set up so that the low 8 values need special handling
    # but all of the high 8 values directly index registers we want to use

    # start out by figuring out what to do with the low values

    zero = const(0)
    one = ~zero

    ram_indirect.track("ram indirect")

    direct_indexing = Bits(which_loadstore.value + [one]*4).clone()

    zero = const(0)
    one = ~zero
    
    regorreg2 = Bits([insn_stream.value[0] ^ which_loadstore.value[0]] + insn_stream.value[1:4] + [zero, one, one, one]).clone()
    p0p1 = Bits([which_loadstore.value[0].clone()] + [const(0) for _ in range(4)] + [~zero for _ in range(3)])
    

    which_reg_ = [
        p0p1,
        p0p1,
        Bits(ram_indirect.value[:7] + [zero]).clone(), # memory lookup
        Bits(insn_stream.value[:2] + [one] + ram_indirect.value[4:7] + [zero, one]).clone(), # status reg lookup

        regorreg2,
        regorreg2,
        p0p1,
        p0p1,
        
        direct_indexing,
        direct_indexing,
        direct_indexing,
        direct_indexing,
        direct_indexing,
        direct_indexing,
        direct_indexing,
        direct_indexing
    ]

    which_reg_ = Bits([do_mux16(which_loadstore,
                             *[x.value[i] for x in which_reg_]) for i in range(8)])
    

    which_reg_.track("which_reg")
    which_reg.connect(Bits(which_reg_.value))

    
    stack_in = Bits([Signal() for _ in range(12)])
    
    stack = [stack_in]

    spacer = Bits([Signal() for _ in range(12)])
    spacer.connect(spacer)
    
    for _ in range(3):
        clock_inner = clock_inner.clone()
        stack.append(dff(stack[-1], clock_inner, 12))

    stack_in, stack_a, stack_b, stack_c = stack
    stack_a.track("stack_a")
    stack_b.track("stack_b")
    stack_c.track("stack_c")

    ###print(", ".join(["v2[%d-3]"%x for x in addr.uids() + u_op.uids()[:6] + addr_sub.uids() + acc.uids()[:4] + [carry.uid[1]] + stack_a.uids() + stack_b.uids() + stack_c.uids() + sum([x.uids()[:4] for x in regs[-32:-16]], [])]))
    

    # todo this can be more efficient by just anding allow_high_bits into stack_a directly

    # We're allowed to use the to 12 only on regs 26-31, for PC_i and TMP/RAM_ADDR
    allow_high_bits = which_reg.value[4] & which_reg.value[3] & (which_reg.value[2] | (which_reg.value[1]))
    #allow_high_bits.track("allow_high_bits")

    is_carry_reg = which_reg.value[7] & which_reg.value[6] & which_reg.value[5] & which_reg.value[4] & which_reg.value[3] & ~which_reg.value[2] & ~which_reg.value[1] & which_reg.value[0]
    
    #mask = Bits([const(1)]*12)
    #for i in range(4, 12):
    #    mask.value[i] = allow_high_bits
    masked = Bits(stack_a.value[:1] +
                  (Bits(stack_a.value[1:4]) & ~is_carry_reg).value +
                  (Bits(stack_a.value[4:]) & allow_high_bits).value)
    
    # If we're writing to a normal register then we only want to write the top 4 bits to the register
    # But if we're writing to the PC then we need to keep the full 12 bits
    reg_write_data.connect(masked)
    (reg_write_data).track("reg_write_data")

    #jump_target.connect(stack_a)
    

    (alu_op|alu_op).track("alu op")

    #any_set_stack1 = Bits([stack_a.value[0] | stack_a.value[1] | stack_a.value[2] | stack_a.value[3]]*12)
    any_set_acc = any_set(acc)
    
    trap_forward = Signal()

    # First let's check if we're a ROM write, ROM read, RAM read, or JCN
    trap_possible = (u_op.value[5] & u_op.value[4] & ~u_op.value[3] & u_op.value[1])
    # This tells us if we're either a RAM or a ROM operation
    trap_ramrom = u_op.value[2] | u_op.value[0]
    # This tells us if the TEST pin for the JCN flag is set
    trap_jcn_flag_set = insn_stream.value[0]
    trap = trap_possible & (trap_ramrom | trap_jcn_flag_set)
        
    trap_forward.connect(trap)

    really_trap = trap & ~trap_forward
    really_trap.track("Trap really")
    #exit(0)
    maybe_color = Signal()

    trap_is_write = u_op.value[2].clone()
    trap_rom_or_ram = u_op.value[0].clone()
    #test_flag = Signal()
    #test_flag.connect(test_flag)
    which_reg = Bits(regs[RAM_REG].value[4:8])
    data = Bits(acc.value[3:4]).clone()

    # add an has_any_data_to_print bit here. check it in the c.
    # I commented out the data things below, see if all the things work still.

    pressed_button = Bits([Signal() for _ in range(12)])
    pressed_button.connect(pressed_button)

    from_rom = Bits([Signal() for _ in range(12)])
    #from_rom.connect(from_rom)
    #print('fr', from_rom.uids())
    #exit(0)

    #data = Bits(acc.value[3:4]).clone()
    #data = Bits(acc.value[3:4]).clone()
    #data = Bits(acc.value[3:4]).clone()
    
    #force_update = Signal()
    #force_update.connect(force_update)

    flag_test = Signal()
    
    #from_rom = from_rom | force_update
    
    jump_invert = insn_stream.value[3]
    jcn_do_jump = (((~any_set_acc) & insn_stream.value[2]) |
                   (carry & insn_stream.value[1]) |
                   (~flag_test & insn_stream.value[0])) ^ jump_invert

    #(test_flag).track("Test_flag")
    #(test_flag & insn_stream.value[0]).track("Test anded")
    
    #carry.track("Carry")
    #(any_set_acc|any_set_acc).track("Any set")
    jcn_do_jump.track("DO JCN")


    # TODO SPACE 1: can break kbp, only need to test a&b
    def kbp(a,b,c,d):
        err = a&b | a&c | a&d | b&c | b&d | c&d
        return [a | c | err,
                b | c | err,
                d | err,
                err,
                zero]

    b1_zero = const(0)
    b2_one = ~b1_zero
    b3_one = b2_one.clone()
    b4_zero = ~b3_one
    b5_zero = b4_zero.clone()
    six = Bits([b1_zero, b2_one, b3_one, b4_zero, b5_zero])
    acc_plus_six = add(Bits(acc.value[:5]), six, zero)

    '''
    zero = const(0)
    one = ~zero
    b0 = acc.value[0]
    b1 = acc.value[1] ^ one
    cc = acc.value[1] & one
    b2 = (acc.value[2] ^ one) ^ cc
    cc = acc.value[2] | cc
    b3 = acc.value[3] ^ cc
    cc = acc.value[3] & cc
    b4 = acc.value[4] ^ cc

    acc_plus_six = Bits([b0, b1, b2, b3, b4]).clone()
    acc_plus_six.track("accplussix")
    '''
    
    do_daa = acc_plus_six.value[4] | carry
    daa = mux(do_daa, acc_plus_six, acc)
    daa = Bits(daa.value[:4] + [(daa.value[4] | carry)] + [zero]*7).clone()
    

    z = Bits([const(0)]*12)

    # TODO unify writing to test_flag with from_rom to the stack

    # TODO actually use the first 8 values of this
    """
    zz = [
        z,z,z,z,z,z,z,z,

        Bits([alu_op.value[0]^x for x in stack_a])&alu_op.value[1], # either x or ~x
        add(stack_a, stack_b, mux(alu_op.value[1], carry, const(0)) ^ alu_op.value[0]),
        Bits(mux(u_op.value[1],
                 mux(u_op.value[0], daa, Bits(kbp(*stack_a.value[:4]))),
                 Bits(stack_b.value[4:8] + [zero])
                 ).value + [const(0)]*7).clone(),
        mux(mux(alu_op.value[1], stack_a.value[0]^alu_op.value[0], ~regs[TMP_REG].value[4]), # MUX/MUXNOT/MUXTMP
            stack_b,
            stack_c),
        mux(u_op.value[1],
            from_rom,
            Bits(stack_a.value[:8] + addr.value[8:]).clone(),
            ),
        Bits([jcn_do_jump]*12),
        Bits((stack_b & u_op.value[1]).value[:4] + stack_a.value[:8]).clone(),
        add(stack_a, stack_a&alu_op.value[0], ~alu_op.value[0])
    ]
    """

    add_result = add(stack_a, stack_b, mux(alu_op.value[1], carry, const(0)) ^ alu_op.value[0])
    add_two = add(stack_a, stack_a&alu_op.value[0], ~alu_op.value[0])

    mux_result = mux(mux(alu_op.value[1], stack_a.value[0]^alu_op.value[0], ~regs[TMP_REG].value[4]), # MUX/MUXNOT/MUXTMP
            stack_b,
            stack_c)

    stack_a_plus = Bits((stack_b & u_op.value[1]).value[:4] + stack_a.value[:8]).clone()

    full_jcn = Bits([jcn_do_jump]*12)
    
    zz = [
        z,
        Bits([alu_op.value[0]^x for x in stack_a]),

        add_result,
        add_result,

        Bits(stack_b.value[4:8] + [const(0)]*8).clone(),
        mux(u_op.value[0], daa, Bits(kbp(*stack_a.value[:4])+[const(0)]*8).clone()),

        mux_result,
        mux_result,
        
        Bits(stack_a.value[:8] + addr.value[8:]).clone(),
        from_rom,
        
        full_jcn,
        full_jcn,

        stack_a_plus,
        stack_a_plus,

        add_two,
        add_two
    ]
    #"""

    alu_output = Bits([do_mux16(Bits(u_op.value[1:5]),
                             *[x.value[i] for x in zz]) for i in range(12)])

    

    alu_output.track("alu output")
    
    next_stack_in = mux(do_alu, alu_output,
                        mux(do_load, loaded_value,
                            stack_a))
    next_stack_in.track("next stack in")
    
    stack_in.connect(next_stack_in)



    # TODO: can instead get the next character exactly when it's asked for
    # by looking at the memory accesses of the CPU

    # op counter tracks the the interacts with the outside world
    op_counter = Bits([Signal() for _ in range(12)])

    # Increment the counter of the number of JCNs we've hit
    inc_op_counter = inc(op_counter, True)

    # Except: when the counter is equal to (13 * 128) we set it back to zero
    # this is because we're using this counter to track the "print head"
    # and it only has thirteen possible entries, so it loops back after 13
    
    counter_mod_thirteen = inc_op_counter.value[7] & inc_op_counter.value[9] & inc_op_counter.value[10]
    inc_op_counter = inc_op_counter & ~counter_mod_thirteen

    op_counter.connect(dff(inc_op_counter, trap & ~trap_ramrom & ~clock_inner, 12))

    print_row = Bits(op_counter.value[7:])
    
    (op_counter).track("op_counter")

    needs_getchar = op_counter.value[6]

    flag_test.connect(needs_getchar)

    printer_clock = Signal()
    keyboard_clock = Signal()
    
    rom_output = dff(acc, trap & u_op.value[0] & u_op.value[1] & u_op.value[2] & ~clock_inner, 12)
    rom_output.track("Rom output")

    printer_shift = [rom_output.value[1]]
    #printer_shift = [rom_output.value[1]]
    #(rom_output.value[2]|rom_output.value[2]).track("Clocked")
    #(rom_output.value[1]|rom_output.value[1]).track("Input")
    #dff(rom_output.value[1], rom_output.value[2], None).track("??")

    # The shift clock needs to be delayed by 1 cycle so the data arrives first
    printer_clock.connect(rom_output.value[2])
    keyboard_clock.connect(rom_output.value[0])
    
    for _ in range(21):
        printer_clock = printer_clock.clone()
        printer_shift.append(dff(printer_shift[-1], printer_clock))
    printer_shift = Bits(printer_shift[1:]).clone()

    (printer_shift).track("Print Shift")
    
#    keyboard_shift = [rom_output.value[1]]
#    for _ in range(21):
#        keyboard_clock = keyboard_clock.clone()
#        keyboard_shift.append(dff(keyboard_shift[-1], keyboard_clock))
#
#    keyboard_shift = Bits(keyboard_shift[1:]).clone()

#    (keyboard_shift).track("Keyboard Shift")

    keyboard_counter = Bits([Signal() for _ in range(12)])
    # if we push in a 1 then this advances the 0 position
    # if we push in a 0 then reset the 0 position to 0
    # TODO: keyboard shift no longer needed
    keyboard_counter.connect(dff(mux(rom_output.value[1],
                                     inc(keyboard_counter),
                                     Bits([zero]*12)),
                                 keyboard_clock,
                                 12))

    ram = dff(acc, trap & ~u_op.value[0] & u_op.value[1] & u_op.value[2] & ~clock_inner, 8)
    ram.track("RAM")
    maybe_color.connect(ram.value[0])

    clear_print_back = Signal()
    clear_print = trap & trap_is_write & ~trap_rom_or_ram & data.value[0]
    clear_print_back.connect(clear_print)

    
    which_print = []

    # TODO: reading from rom should be to the register file, not a special new port

    # print_row is the current row that the print head is on, and ready to strike
    
    # If we have clear_print set then we've called for a new line
    # This means we need to clear the old printed characters
    store_print_row = inc(print_row) & ~clear_print
    store_print_row.track("print row")

    printer_shift = ((printer_shift & ram.value[1]) | clear_print_back)
    
    open("extra.h", "w").write(f"#define store {store_print_row.value[0].uid[1]}\n#define printer_shift {printer_shift.value[0].uid[1]}\n#define op_counter {op_counter.value[0].uid[1]}\n#define clear_back {clear_print_back.uid[1]}\n#define clear {clear_print.uid[1]}\n")
    
    for i in range(20):
        out = dff_circ(store_print_row, printer_shift.value[i])
        #prior = Bits([Signal() for _ in range(5)])
        #out = dff_circ(mux(printer_shift.value[i], store_print_row, prior), clock_inner)
        #prior.connect(out)
        which_print.append(out)

    which_print = Bits(sum([x.value for x in which_print], []))
    which_print.track("Printing")
    print('AAA', which_print.value[0].uid)
    print('AAA', ram.value[0].uid)
    
    pressed_button_read = (pressed_button ^ clock_inner) ^ clock_inner # force updates here

    #print(pressed_button_read.uids())

    # We press one button at a time, and the ID of that button is here
    # We need to only activate it when the read head comes around to it
    # which happens when the 0 bit in keyboard_shift is equal to pressed_button[4:8]
    pressed_button_masked = pressed_button_read & ~any_set(Bits(keyboard_counter.value[:4]) ^ Bits(pressed_button_read.value[4:8]))

    other = Bits([print_row.value[3] & print_row.value[2] & ~print_row.value[1] & ~print_row.value[0], const(0), const(0), const(0)])

    rom_holder = mux(which_reg.value[0],
                       Bits(pressed_button_masked.value[:4]),
                       other
                     )
                     
    from_rom.connect(Bits(rom_holder.value + [zero]*8))


    addr_ = addr.clone()

    # add for NOPs for space
    # This is because addr is a 12 bit value
    # but insn_stream_ is only 8 bits
    # so to make the offsets line up we want 12 NOPs here
    # (Because of the DFF for insn_stream it adds an extra 8, so we have 8+8-12-4=0)
    [const(0) for _ in range(4)]
    
    insn_stream_before = rom(addr_, None, load_insn_stream, 8, force=fake, logger=logger)

    insn_stream_before_.connect(insn_stream_before)
    
    (insn_stream_before).track("insn_stream_before")
    
    insn_stream_ = dff(insn_stream_before, clock_outer, 8)
    
    (insn_stream_).track("insn_stream")
    #(next_byte_).track("next_byte")
    
    u_ops_ = rom(insn_stream_, None, table, 128, force=fake, logger=logger)
    u_ops_ = Bits(u_ops_.value[:96])

    #print('u_ops_', u_ops_.uids())
    
    (u_ops_).track("u_ops")

    insn_stream.connect(insn_stream_)
    u_ops_base.connect(u_ops_, reuse=True)
    
    # rom[2] hold print_row
    # rom[1] holds pressed_button

    lines_to_write = [
        "#define TT " + str(len(clock_outer.signals)),
#        "#define needs_getchar_uid " + str(needs_getchar.uid[1]),
        "#define trap_id " + str(really_trap.uid[1]),
#        "#define printer_shift_uid " + str(printer_shift.uids()[0]),
#        "#define keyboard_shift_uid " + str(keyboard_shift.uids()[0]),
#        "#define keyboard_counter_uid " + str(keyboard_counter.uids()[0]),
        "#define print_uid " + str(which_print.uids()[0]),
#        "#define print_row_uid " + str(print_row.uids()[0]),
#        "#define rom_holder " + str(rom_holder.uids()[0]),
#        "#define rom_enabled " + str(rom_enabled.uid[1]),
#        "#define which_uid " + str(which_reg.uids()[0]),
#        "#define pressed_button_masked " + str(pressed_button_masked.uids()[0]),
#        "#define pressed_button_uid " + str(pressed_button.uids()[0]),
    ]
    #print("\n".join(lines_to_write))
    
    ## Open const.h file for writing
    #with open('const.h', 'w') as const_file:
    #    # Write each line to the file
    #    for line in lines_to_write:
    #        const_file.write(line + '\n')

    open("debug.h", "w").write(
        (", ".join(["v2[%d]"%x for x in addr.uids() + acc.uids()[:4] + [carry.uid[1]] + sum([x.uids()[:4] for x in regs[-32:-16]], [])]) + ",") +
        (", ".join(["v2[%d]"%x for x in sum([x.uids()[:4] for x in regs[:128]],[]) + sum([regs[128 + 4 + i + (j<<3)].uids()[:4] for j in range(8) for i in range(4)],[])]) + ",") +
        (", ".join(["v2[%d]"%x for x in op_counter.uids()])))
    

    return clock_outer, addr, regs, CIRC_PATTERNS, (len(clock_outer.signals), really_trap.uid[1], which_print.uids()[0])

if __name__ == "__main__":
    aa = time.time()

    from test4004 import run
    run("test_daa", debug=True)
    exit(0)
    
    c, addr, regs, CIRC_PATTERNS = make_4004("""
        JCN 9, 0   ; just increment the test counter
        LDM 6 ; write 1
        WRR
        LDM 0 ; down
        WRR
        LDM 4 ; write 1
        WRR
        LDM 0 ; down
        WRR
        LDM 6 ; write 1
        WRR
        LDM 0 ; down
        WRR
        LDM 6 ; write 1
        WRR
        LDM 0 ; down
        WRR

        LDM 2
        WMP
    """)
    
    #c, addr, regs, CIRC_PATTERNS = make_4004("""
    #A: IAC
    #JUN A
    #""")

    def get_hist(i):
        if True:
            print()
            building = None
            r = ""
            for x in Signal.signals:
                if x.name:
                    #print(x.name, x.val())
                    if len(x.name) > 4 and x.name[-4] == '.':
                        if building == x.name[:-4]:
                            r = str(x.val()) + r
                        else:
                            if building is not None:
                                print(building, r)
                            r = str(x.val())
                            building = x.name[:-4]
                            #print("Start building", building)
                    else:
                        if building is not None:
                            print(building, r)
                        building = None
                        print(x.name, x.val())
            print(building, r)
            print(regs[-32:-16])
            print(regs[-16:])
            print(regs[:-32])
    
    #c, addr, regs, CIRC_PATTERNS = make_4004("""
    #IAC
    #ADD 0
    #""")
    

    def fn(x):
        print(addr)

    def write(i):
        for x in c.signals:
            if x.value:
                print(x.uid[1]+2, end=' ')
        exit(0)
        

    sim(10000, c, cb=get_hist)#, cb2=write)
    
    exit(0)


    signals = c.signals

    compressed = verilog.compress(signals, CIRC_PATTERNS)
    compressed = verilog.compress_repeats(compressed)
    
    byte_compressed = str([y for x in compressed for y in x]).replace(" ","")
    print('len',len(gzip.compress(bytes(byte_compressed, 'ascii')))) 
    

