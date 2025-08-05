import time
import itertools
import inspect
import numpy as np
import re
import gzip
import fastlz
import collections
import heapq
import sys

if len(sys.argv) >=2 and sys.argv[1] == "quet":
    DEBUG = False
else:
    DEBUG = False

MAX = 0x1

# TODO remove the extra bytes that are still around after generating the reuse=True.
# To do this decrement the index of all bytes after that counter.

# TODO better to store the number of arguments to fucntions once instead of repeating every time I call
# maybe not? Because with LZ it's always prefixed that way anyway.

# TODO don't make function calls use the absolute position.

# TODO get rid of repeat when it's non-incrementing


#SPECIAL_LAST = 100000
#SPECIAL_INC  = 100001
#SPECIAL_ARGS = 100002
SPECIAL_ARGS = int(2e5)


class Signal():
    signals = []
    signalmap = {}
    needs_tick = []

    def __init__(self, fn=lambda: 0, how=None, add_signal=True, value=0):
        global UID
        if add_signal:
            if len(self.signals) > 800000: raise
            self.signals.append(self)
            self.added_signal = True
        else:
            self.added_signal = False
        self.value = value
        self.fn = fn
        self.uid = ('VAR', len(self.signals)-1)
        self.how = how
        self.name = None
        self.drop = None
        self.alert = None
        self.depends = []
        if self.how is not None:
            if self.how[0] == 'VAR':
                pass
            else:
                if self.how != 'BAD':
                    for x in self.how[1:]:
                        x.depends.append(self)
                


        if False:
            self.LINE = []
            for callerframerecord in inspect.stack()[1:]:
                frame = callerframerecord[0]
                info = inspect.getframeinfo(frame)
                self.LINE.append(info.filename.split("/")[-1] + ":" + str(info.lineno))
            print("Have", self.LINE)

        if False:
            frame = inspect.currentframe().f_back
            info = inspect.getframeinfo(frame)
            self.LINE = info.filename.split("/")[-1] + ":" + str(info.lineno)

#    def __getattr__(self, v):
#        if v == 'value':
#            o = self.signals[self.uid[1]]._value
#            if self._value != o: raise
#            return o

    def tick(self):
        prev = self.value
        self.value = self.fn()
        if self.value != prev:
            return self.depends
        return []
        #if self.value is self.alert:
        #    print(self.name, "set to", self.value)
        
    def __and__(a, b):
        return Signal(lambda: a.value & b.value,
                      ('&', a, b))

    def __xor__(a, b):
        return Signal(lambda: a.value ^ b.value,
                      ('^', a, b))
    
    def __or__(a, b):
        return Signal(lambda: a.value | b.value,
                      ('|', a, b))

    def __eq__(self, other):
        return self.uid == other.uid and self.how == other.how

    def __hash__(self):
        if self.how is None: return hash(self.uid)
        hh = hash(tuple(hash(x) if not isinstance(x, Signal) else x.uid for x in self.how))
        return hash((self.uid, hh))
    
    def mux(self, a, b):
        if isinstance(self, Bits) or isinstance(a, Bits) or isinstance(b, Bits):
            longest = 0
            if isinstance(self, Bits): longest = max(longest, len(self.value))
            if isinstance(a, Bits): longest = max(longest, len(a.value))
            if isinstance(b, Bits): longest = max(longest, len(b.value))
            
            if not isinstance(self, Bits):
                self = Bits([self for _ in range(longest)])
            if not isinstance(a, Bits):
                a = Bits([a for _ in range(longest)])
            if not isinstance(b, Bits):
                b = Bits([b for _ in range(longest)])
            return self.mux(a,b)
        return Signal(lambda: (a.value if self.value else b.value),
                      ('MUX', self, a, b))
    
    def __invert__(a):
        return Signal(lambda: MAX ^ a.value,
                      ('~', a))

    def connect(self, other, reuse=False):
        def update():
            return self.signals[other.uid[1]].value
        self.fn = update
        if not reuse:
            self.how = other.uid
            other.depends.append(self)
            self.signals[other.uid[1]].depends.append(self)
        else:
            assert self.added_signal == False
            self.how = other.uid
            self.uid = other.uid

            vs = list(other.depends)
            
            self.depends = other.depends
            self.depends.extend(vs)

    def alert_on(self, x):
        self.alert = x

    def linearize(self):
        return ('=', self.uid, self.how)

    def c_linearize(self):
        def linearize(x):
            if isinstance(x, Signal): x = x.uid
            if x[0] == 'VAR':
                return 'v['+str(x[1])+"] ^ v[0]"
            elif x[0] == 'CONST':
                return 'v['+str(self.uid[1])+"] ^ v[0]"
            elif x[0] in ['&', '|', '^']:
                return 'v['+str(x[1].uid[1]) + '] ' +x[0] +' v['+ str(x[2].uid[1])+"]"
            elif x[0] in ['~']:
                return 'v['+str(x[1].uid[1])+"] ^ v[1]"
            elif x[0] in ['MUX']:
                return 'v[%d] ? v[%d] : v[%d]'%(x[1].uid[1], x[2].uid[1], x[3].uid[1])
            else:
                print(x)
                raise 
                
        return 'v['+str(self.uid[1]) + '] = ' + linearize(self.how)+';'

    def match(self, other, assigns, aoff, boff, fuzzy=True):
        #print()
        print("Verify", self, other, aoff, boff)
        #print(assigns)
        def ok(a, b):
            print('check',a,b, type(a))
            if isinstance(a, Signal):
                if fuzzy:
                    if not isinstance(b, Signal):
                        #print("F1")
                        return False
                    if a.uid[1]-aoff == b.uid[1]-boff:
                        #print("T")
                        return True
                    if b.uid[1] in assigns:
                        #print(assigns)
                        #print("Check", b.uid[1], a.uid[1])
                        return assigns[b.uid[1]] == a.uid[1]
                    else:
                        #print("Set", b.uid[1], a.uid[1])
                        assigns[b.uid[1]] = a.uid[1]
                        return True
                else:
                    return b[0] == 'VAR' and a.uid-aoff == b.uid-boff
            if a[0] == 'VAR' and type(b) == tuple and b[0] == 'VAR' and fuzzy:
                if a[1]-aoff == b[1]-boff:
                    return True
                if b[1] in assigns:
                    #print(assigns)
                    #print("Check", b[1], a[1])
                    return assigns[b[1]] == a[1]
                else:
                    #print("Set", b[1], a[1])
                    assigns[b[1]] = a[1]
                    return True
                
            if a[0] == 'TRACE':
                if a in assigns:
                    return assigns[a] == b
                else:
                    assigns[a] = b
                    return True
            if b[0] == 'TRACE':
                if b in assigns:
                    #print('check', assigns[b], a)
                    return assigns[b] == a
                else:
                    #print("Set", b, a)
                    assigns[b] = a
                    return True
            if b[0] == 'VAR':
                return a[0] == 'VAR' and a[1]-aoff == b[1]-boff
            if a[0] == 'CONST' or b[0] == 'CONST':
                return a == b
            if a[0] != b[0]:
                return False
            return all(ok(q, w) for q,w in zip(a[1:], b[1:]))
                            
        return ok(self.how, other.how)

    def compress(self):
        offset = self.uid[1]

        out = []
        #print('iam', self.uid, self.how, self.name, self.LINE)

        def encode(c):
            return [(c, 2**16)]
            BITS = 6
            if 0 <= c+2**(BITS-1) < 2**BITS:
                return [(0, 2), (c+2**(BITS-1), 2**BITS)]
            BITS = 12
            assert 0 <= c + 2**(BITS-1) < 2**BITS
            return [(1, 2), (c + 2**(BITS-1), 2**BITS)]
        
        def write(x):
            if isinstance(x, Signal): x = x.uid
            if x is None:
                print("Fail on", self, self.LINE)
                raise
            if x[0] == 'VAR':
                return ((0, 4), *encode(offset-x[1]))
            if x[0] == 'CONST':
                return ((0, 4), (0, 0))
            ops = ['~']
            if x[0] in ops:
                return ((1, 4), *write(x[1])[1:])
                #return ((3, 4), (2, 0), *write(x[1])[1:], (0,0), (1,0))
            ops = ['|', '&', '^']
            if x[0] in ops:
                return ((2+ops.index(x[0]), 4), *write(x[1])[1:], *write(x[2])[1:])
            ops = ['MUX']
            if x[0] in ops:
                return ((5, 4),
                        *write(x[1])[1:], *write(x[2])[1:], *write(x[3])[1:])
            if x[0] == 'TRACE':
                return (None, (SPECIAL_ARGS+x[1], 0))
            raise

        o = write(self.how)

        #print(self.uid, [x[0] for x in o])
        return [x[0] for x in o]

        ans = 0
        for a,b in o:
            assert 0 <= a < b
            base = int(np.ceil(np.log(b)/np.log(2)))
            assert 2**base >= b
            ans <<= base
            ans |= a
        #print(ans, o)
        return ans
        

    def __repr__(self):
        if self.how is None: return 'HNone'
        if self.how == -1: return str(self.uid)
        how = str(tuple([x.uid if isinstance(x,Signal) else str(x) for x in self.how]))
        return '[' + str(self.uid) + ' = ' + str(how) + ']'# + " ;; " + str(self.value)

    def val(self):
        return str(self.value)

    def track(self, name):
        if self.name is not None: raise
        self.name = name

    def clone(self):
        s = Signal()
        s.connect(self)
        return s
        
class Bits:
    def __init__(self, values):
        self.value = values

    def connect(self, other, reuse=False):
        assert len(other.value) == len(self.value)
        [x.connect(y, reuse) for x,y in zip(self.value, other.value)]

    def __and__(a, b):
        if isinstance(b, Signal):
            return Bits([x & b for x in a.value])
        return Bits([x & y for x,y in zip(a.value, b.value)])

    def __xor__(a, b):
        if isinstance(b, Signal):
            return Bits([x ^ b for x in a.value])
        return Bits([x ^ y for x,y in zip(a.value, b.value)])
    
    def __or__(a, b):
        if isinstance(b, Signal):
            return Bits([x | b for x in a.value])
        return Bits([x | y for x,y in zip(a.value, b.value)])

    def __eq__(a, b):
        return Bits([x == y for x,y in zip(a.value, b.value)])

    def __add__(a, b):
        return Bits([x + y for x,y in zip(a.value, b.value)])

    def mux(self, a, b):
        return Bits([x.mux(y,z) for x,y,z in zip(self.value, a.value, b.value)])
    
    def __invert__(a):
        return Bits([~x for x in a.value])

    def __iter__(self):
        return iter(self.value)

    def clone(self):
        out = Bits([Signal() for _ in range(len(self.value))])
        out.connect(self)
        return out

    def __repr__(self):
        return "%01x"%int(''.join([str(x.value) for x in self.value][::-1]),2)

    def val(self):
        return "%01x"%int(''.join([str(x.value) for x in self.value][::-1]),2)

    def uids(self):
        return [x.uid[1] for x in self.value]
    
    def track(self, name):
        [x.track("%s.%03d"%(name,i)) for i,x in enumerate(self.value)]
    
constcache = {}
def const(n, depth=None):
    assert depth is None
    if n == 0:
        s = Signal()
        s.connect(s)
        return s
    else:
        #s = Signal()
        #s.how = ("VAR", -1)
        #s.value = 1
        #s.fn = lambda: 1
        s = Signal()
        s.connect(s)
        return ~s

    #if depth is None:
    #    if n not in constcache: 
    #        constcache[n] = Signal(lambda: n, value=n)
    #        if n == 1:
    #            constcache[n]._value = 1
    #    return constcache[n]
    #else:
    #    return Bits([const((n>>i)&1) for i in range(depth)])

class Trace(Signal):
    def __init__(self, n, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.signals.pop()
        self.uid = ('TRACE', n)
        self.how = ('TRACE', n)

CIRC_PATTERNS = {}

def circ(*bit_depth):
    def out(fn):
        global constcache
        if inspect.getargspec(fn).varargs is None:
            assert len(inspect.getargspec(fn).args) == len(bit_depth)

        name = inspect.getargspec(fn)
        #print('circ', name)

        args = []
        i = 0
        for bit in bit_depth:
            if bit is None:
                args.append(0)
            elif bit == 1:
                args.append(Trace(i))
                i += 1
            else:
                args.append(Bits([Trace(i+j) for j in range(bit)]))
                i += bit

        UID = len(CIRC_PATTERNS)

        out = fn(*args)
        if isinstance(out, Bits):
            out = out.value[0]
        CIRC_PATTERNS[fn.__name__] = list(out.signals)
        while len(out.signals) > 0:
            out.signals.pop()


        def fn2(*args):
            assert len(args) == len(bit_depth)
            if isinstance(args[0], Bits):
                signals = args[0].value[0].signals
            else:
                signals = args[0].signals
            size = len(signals)
            o = fn(*args)
            for s in signals[size:]:
                s.drop = True

            uids = []
            for x in args:
                if isinstance(x, Signal):
                    uids.append(x)
                elif isinstance(x, Bits):
                    uids.extend(z for z in x.value)
                else:
                    raise
            #s.drop = (4, UID, len(uids), *uids)

            def compute_uids():
                actual_uids = [x.uid[1]-signals[size].uid[1] for x in uids]
                return (6, UID, len(actual_uids), actual_uids[0], *np.diff(actual_uids))
            
            s.drop = compute_uids
            
            return o
        return fn2
    return out

def sim(steps, c=None, cb=lambda i: 0, cb2=lambda i: 0):
    a = time.time()
    last_c = 0
    hist = []

    needs_tick = [x.uid[1] for x in c.signals]
    heapq.heapify(needs_tick)

    #s = set(x.uid[1] for x in c.signals)
    #assert sorted(s) == list(range(len(s)))
    
    values = [0]*len(c.signals) + [0, 1]

    did_rom_many = []
    
    for i in range(steps):
        #for x in c.signals:
        #    x.tick()
        next_tick = []
        next_did = {}
        did = {}
        did_update = {}
        current = -1
        did_rom = 0

        #do_all = False
        #if i%(16) > 14:
        #    needs_tick = list(range(len(c.signals)))
        #    heapq.heapify(needs_tick)
        #    do_all = True

        #print("Step", i)
        while len(needs_tick):
            x = heapq.heappop(needs_tick)
            if x in did: continue
            did[x] = True
            current = x
            #if current > 25000 and not do_all:
            #    continue
            x =  c.signals[x]

            if current > 26000:
                did_rom += 1
            
            prior = x.value
            update = x.depends
            
            method, *rest = x.how


            if method == 'VAR':
                values[x.uid[1]] = values[rest[0]]
            elif method == '~':
                values[x.uid[1]] = 1-values[rest[0].uid[1]]
            elif method == '&':
                values[x.uid[1]] = values[rest[0].uid[1]] & values[rest[1].uid[1]]
            elif method == '|':
                values[x.uid[1]] = values[rest[0].uid[1]] | values[rest[1].uid[1]]
            elif method == '^':
                values[x.uid[1]] = values[rest[0].uid[1]] ^ values[rest[1].uid[1]]
            elif method == 'MUX':
                values[x.uid[1]] = values[rest[1].uid[1]] if values[rest[0].uid[1]] else  values[rest[2].uid[1]]
            else:
                print(method)
                raise
            x.value = values[x.uid[1]]

            if x.value == prior:
                continue
            #print("Maybe", x.uid)

            if x.alert:
                x.alert()
            
            did_update[x] = True
            for o in update:
                if o.uid not in did and o.uid[1] > current:
                    heapq.heappush(needs_tick, o.uid[1])
                if o.uid not in next_did:
                    #print('next time add', o.uid)
                    next_did[o.uid] = True
                    heapq.heappush(next_tick, o.uid[1])

            #allowed_update = i%16 > 0
            #if not allowed_update:
            #    print("Nope")
            #    x.value = values[x.uid[1]] = prior
                
            #print(did_rom)

        #if did_rom:
        #    print(i%16, did_rom)
        #if did_rom > 0:
        #    print(i%(16*16),did_rom)
        #    did_rom_many.append(i)
        needs_tick = next_tick
        next_did = {}

        #print(len(did))
        #print("Echo uop=%d%d%d%d%d%d\n"%(values[24964], values[24979], values[24994], values[25009], values[25024], values[25039]))
        #if i%16 == 0:
            #print('aa', [values[x] for x in [22961, 22962, 22963, 22964, 23024, 23025, 23026, 23027, 23087, 23088, 23089, 23090, 23150, 23151, 23152, 23153, 23213, 23214, 23215, 23216, 23276, 23277, 23278, 23279, 23339, 23340, 23341, 23342, 23402, 23403, 23404, 23405, 23465, 23466, 23467, 23468, 23528, 23529, 23530, 23531, 23591, 23592, 23593, 23594, 23654, 23655, 23656, 23657, 23717, 23718, 23719, 23720, 23780, 23781, 23782, 23783, 23843, 23844, 23845, 23846, 23906, 23907, 23908, 23909]])
        #print(" ".join(map(str,list(np.where(values)[0]))),end=" \n")

        
        cb2(i)
        #exit(0)
        if c is None or (c.value == 0 and c.value != last_c):
            cb(i)
        if c is not None:
            last_c = c.value
    print('update', collections.Counter(np.array(did_rom_many)%(16*16)))
            


def compute_time_to_stabilize(c):
    prev = [tuple([x.value for x in signals])]
    time = [0]
    ok = [False]
    
    def fn_tick(i):
        cur = tuple([x.value for x in signals])

        if c.value == 0:
            time[0] = i
            ok[0] = True

        if c.value > 0 and sum(a!=b for a,b in zip(cur, prev[0])) == 1 and ok[0]:
            print(i, i-time[0])
            ok[0] = False

        prev[0] = cur
        
    sim(4000, c, lambda i: 0, fn_tick)
    exit(0)

def compress(signals, CIRC_PATTERNS):
    compressed = [[]]

    for i,v in enumerate(CIRC_PATTERNS.values()):
        for x in v:
            compressed[-1].append(x.compress())
        compressed[-1].append([10])
        compressed.append([])
    
    for x in signals:
        if x.drop is None:
            # normal commmand
            cc = x.compress()
            #print(cc, x.uid, x.LINE)
            compressed[-1].append(cc)
        elif x.drop is True:
            pass
        else:
            # substitute with something else instead
            compressed[-1].append(x.drop())
    return compressed

"""
def compress_repeats(compressed):
    compressed2 = []
    while len(compressed) > 0:
        print(len(compressed))
        #print(compressed[0])
        def ok(x):
            y = compressed[0]
            #if y[0] == 4: return False
            
            if len(x) != len(y): return False
            return True

        keep = list(itertools.takewhile(ok, compressed))
        keep = np.array(keep)
        print("maybe", len(keep))
        
        while keep.dtype == np.int64 and len(keep) > 2:
            idx = np.where(np.any(keep[0] != keep[1:],0))[0]
            #print("II", idx)
            counting = keep[:,tuple(idx)]
            gap = counting[:-1] - counting[1:]
            #print(compressed[0])
            #print(gap)

            if np.all(gap == gap[:1,None]):
                compressed2.append(compressed[0])
                compressed2.append((5, len(keep), len(idx), *idx, *-gap[0]))
                compressed = compressed[len(keep):]
                break
            else:
                keep = keep[:-1]
                keep = np.array(keep)
        else:
            compressed2.append(compressed[0])
            compressed = compressed[1:]
                
            
    return compressed2
"""

def compress_repeats(compressed):
    compressed2 = []
    while len(compressed) > 0:
        #print(compressed[0])

        keep = [compressed[0]]
        for i in range(1,len(compressed)):
            if len(compressed[i]) == len(compressed[0]):
                keep.append(compressed[i])
            else:
                break
        
        keep = np.array(keep)
        
        if len(keep) > 2:
            all_gaps = keep[1:] - keep[:-1]
            wanted_gap = all_gaps[0]
            ok = np.all(wanted_gap == all_gaps,1)

            first_bad = np.where(~ok)[0]
            if len(first_bad) > 0:
                keep = keep[:first_bad[0]+1]

            idx = np.where(wanted_gap)[0]

            if len(keep) > 2:
                # if we're doing more than 100 repeats (and then LZ would become really slow)
                # or if we're doing any incrementing and as long as we're not making things bigger
                #  by storing the incrementing
                if len(keep) > 1 or len(idx) > 0:
                    compressed2.append(compressed[0])
                    if len(idx) > 0:
                        compressed2.append((7, len(keep), len(idx), idx[0], *np.diff(idx), wanted_gap[idx][0], *np.diff(wanted_gap[idx])))
                        #bitmask = np.sum(2**np.array(idx))
                        #compressed2.append((7, len(keep), bitmask, wanted_gap[idx][0], *np.diff(wanted_gap[idx])))
                    else:
                        compressed2.append((7, len(keep), len(idx)))
                else:
                    compressed2.extend(compressed[:len(keep)])
                compressed = compressed[len(keep):]
            else:
                compressed2.append(compressed[0])
                compressed = compressed[1:]
        else:
            compressed2.append(compressed[0])
            compressed = compressed[1:]
                
            
    return compressed2

#print(my_compress([99, 103, 107, 99, 103, 107, 99, 103, 107]))
#exit(0)

if __name__ == "__main__":

    rom_start = []

    #@circ(1)
    #def foo(x):
    #    return x | x | x 

    #c = const(0)
    #foo(c)
    

    from risc import make_4004, ACC_REG
    
    c, addr, regs, CIRC_PATTERNS, special_constants = make_4004("""
    IAC
    JUN 0
    WRR
    """, fake=True, logger=rom_start)


    #sim(2000, c, lambda x: print(regs))
    #exit(0)

    """
    c = Signal()
    d = ~c

    e = Signal()
    e.connect(d)

    f = Signal()
    f.connect(e, reuse=True)
    
    c.connect(f)

    out = d ^ f
    
    
    CIRC_PATTERNS = {}
    regs = [Bits([c])]
    ACC_REG = 0
    #"""
    


    txt = []


    
    for i, (offset, toks) in enumerate(rom_start):
        txt.append(offset)
        if i == 0:
            open("run.4004","wb").write(bytes(toks))
        else:
            for z in toks:
                for i in range(16): # maximum of 16 uops
                    txt.append((z>>(i*6))&0x3F)
    #exit(0)

    txt += [ord(x) for x in " 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34 05612  .-78  9.34                   .Mm+-  CR^=  S%x/ #Mm*1  CMTK  EE23"]
    # one extra zero is here for -1 (returned when getchar() gets nothing) to have a 0
    txt += [0]
    #txt += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 50, 0, 49, 72, 34, 104, 100, 84, 68, 98, 82, 66, 97, 81, 65, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 88]
    txt += [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 50, 0, 49, 72, 34, 104, 100, 84, 68, 98, 82, 66, 97, 81, 65, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 33, 0, 0, 2, 113, 0, 0, 0, 0, 120]
    

    def f(i):
        print(1, end=" ")
        for x in c.signals:
            if x.value:
                print(x.uid[1]+2,end=" ")
        print()
    
    
    signals = c.signals

    f = open("/tmp/b.c","w")
    f.write("""#include<stdio.h>\nint main() {\nint v[1000000] = {0};""")
    f.write("v[1] = 1;\n");
    f.write("for (int i = 0; i < 55000; i++) {\n")
    for i,x in enumerate(signals):
        x = x.c_linearize()
        f.write(x+"\n")
    f.write('printf("%d %d %d %d\\n", ' + ", ".join("v["+str(x.uid[1])+"]" for x in regs[ACC_REG].value[:4]) + ');')
    f.write("}\n")
    f.write("}\n")
    f.close()
    
    #exit(0)
    #for x in c.signals:
    #    print(x.linearize());
    


    # Find duplicated gates
    """
    print("Dup gates")
    did = {}
    update = []
    for x in signals:
        if ('VAR', -2) == x.how: continue
        if x.how in did:
            print(x.how)
            update.append((x.uid, did[x.how].uid))
            
            #x.drop = True
            #x.uid = did[x.how].uid
        else:
            did[x.how] = x
    print("End dups", len(update))
    #"""
    

    compressed = compress(signals, CIRC_PATTERNS)

    num_signals = len(c.signals)
    
    compressed = [compress_repeats(x) for x in compressed]
    print("Comp", compressed)
    lens = [sum(map(len,x)) for x in compressed]
    lens = [0] + list(np.cumsum(lens))[:-1]
    # PRINT ID HERE
    #print([type(x) for x in special_constants])
    compressed = [[[*special_constants,len(lens)-1] + [x + len(lens) + 4 for x in lens]]] + compressed
    #compressed = [[[158028,25494,26566,len(lens)] + [x + len(lens) + 4 for x in lens]]] + compressed
    #compressed = compress_increments(compressed)
    compressed = sum(compressed, [])

    #"""


    yy = [y for x in compressed for y in x] + [9]
    yy.extend(txt)

                
                
        

    #zz = []
    #last = -1e9
    #for x in yy:
    #    if len(zz) >= 3 and zz[-3]+3 == zz[-2]+2 == zz[-1]+1 == x:
    #        zz.pop()
    #        zz.append(SPECIAL_INC)
    #        zz.append(SPECIAL_INC)
    #        zz.append(SPECIAL_INC)
    #    elif last+1 == x:
    #        zz.append(SPECIAL_INC)
    #    else:
    #        zz.append(x)
    #yy = zz

    #    last = x

    #print(zz)
    #byte_compressed = np.array(yy, dtype=np.uint32).view(np.uint8).reshape((-1,4))[:,:3].flatten().tobytes()
    byte_compressed = bytes(str(yy).replace(" ",""), 'ascii')

    open("/tmp/b","wb").write(byte_compressed[1:-1].replace(b",",b'/'))

    open("data.h","w").write("int prog_src[] = {" + ",".join(map(str,yy)) + "};")
    
    
    print('len compress',len(gzip.compress(byte_compressed)), 'num signals', num_signals)
    #print(len(my_compress(zz)))
    #exit(0)
    
    #print(my_compress([y for x in compressed for y in x]))

    if DEBUG:
        print("BEGIN COMPRESS")        
        for x in compressed:
            print(", ".join(map(str,x)),',')
        print("END COMPRESS")

    #"""

    #exit(0)
    
    #print('len',len(gzip.compress(bytes(str(["%x"%x.compress() for x in signals]), 'ascii'))))
    
    
    #aa = time.time()
    #print("START")
    #def fn(i):
    #    print(contents)
    #    print()
    #    #print('addr', addr, 'alu', alu_op, 'jump', do_jump.value, 'cjmp', conditional_jump.value, 'inv', invert_condition.value, 'alu out', str(data_wire), 'enable', enable_reg_write.value)
    #sim(200, c, fn)
    #print(time.time()-aa)

    #compute_time_to_stabilize(c)
    
    #exit(0)
    

    
    #for i,x in enumerate(signals):
    #    x = x.linearize()
    #    assert x[1][0] == 'VAR' and i == x[1][1]
    #    eq = x[2]
    #    dump([], eq, i)
    
    
    #v_to_next = {i: [] for i in range(len(signals))}
    #for x in signals:
    #    rest = re.findall("v[0-9]*", x.linearize().split("=")[1])
    #    for var in rest:
    #        v_to_next[int(var[1:])].append(x.linearize())
    #"""
    
    
    #for i in range(len(signals)):
    #    print("int v%d = 0;"%i)
    #for k,v in v_to_next.items():
    #    print("void do_%d() {"%k)
    #    print("    ", "if (did[%d] == TAG) return;"%k)
    #    print("    ", "did[%d] = TAG;"%k)
    #    for x in v:
    #        print("   ", x)
    #        print("   ", "todo[I++] = &do_" + x.split()[0][1:] + ";")
    #    print("}");
    
    
    
