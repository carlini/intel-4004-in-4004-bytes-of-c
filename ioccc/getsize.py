import collections
import re

# TODO replace the unnecessary newlines

lines = open("prog.c").readlines()

if True:
    replace = []
    ls = []
    for line in lines:
        if '// MAKEINLINE' in line:
            line = line.strip()
            assert line.startswith("int")
            var, rest = line[3:].split("=")
            replace.append((var.strip(), rest.split(";")[0]))
        elif '// DEBUGONLY' in line:
            continue
        elif '// MAKETMP' in line:
            if 'TMP2' in line:
                newvar = 'tmp2'
            elif 'TMPP' in line:
                newvar = 'tmpp'
            else:
                newvar = 'tmp'
            line = line.strip()
            assert line.startswith("int")
            var, rest = line[3:].split(" = ")
            ls.append(newvar+ " = " + rest)
            replace.append((var.strip(), newvar))
        else:
            ls.append(line)
    print(replace)
    lines = "\n".join(ls)
    for src,dst in replace:
        lines = lines.replace(src,dst)
    lines = lines.split("\n")



lines = [x.split("//")[0] for x in lines]

cptr = open("lz.h").read()

lines = [x for x in lines if 'char* Z' not in x]
lines = "\n".join(lines)
#lines = open("const.h").read() + "\n" + lines

#lines = lines.replace('#include "const.h"', '')
lines = lines.replace('#include "lz.h"', '')
lines = lines.replace("#define W for(i=0;i<tmp;i++)", "")
lines = lines.replace("#define INC_Q *Q++", "")
lines = lines.replace("%d", "\1")
lines = lines.replace(".h", "\2")

lines = lines.replace("num_change", "fraction")
lines = lines.replace("num_args", "byte_offset")
lines = lines.replace("write_size", "ctx")
lines = lines.replace("memptr", "z")
lines = lines.replace("counter", "top")
lines = lines.replace("ref", "now")
lines = lines.replace("now", "print_row")
lines = lines.replace("int prev_val", "")
lines = lines.replace("prev_val", "fraction")
lines = lines.replace("building_chr", "fraction")
lines = lines.replace("writesize", "ctx")
lines = lines.replace("subtract_it", "byte_offset")
lines = lines.replace("result_ans", "needs_update")
lines = lines.replace("prev_start", "needs_update")
lines = lines.replace("todo", "byte_offset")
lines = lines.replace("pressed_button", "byte_offset")
lines = lines.replace("dat_ptr", "Q")
lines = lines.replace("argc", "counts")
lines = lines.replace("argv", "ctx")


lines = lines.replace('printf("%s",', "printf(")



while '/*' in lines:
    first, _, rest = lines.partition("/*")
    drop, _, rest = rest.partition("*/")
    lines = first+rest

for _ in range(3):
    for _ in range(10):
        lines = lines.replace("\t", " ")
        lines = lines.replace("  ", " ")
    lines = lines.replace("\n ", "\n")
    for _ in range(10):
        lines = lines.replace("\n\n", "\n")
    
    for op in '+-*%^{}();?:,=<>|&[]/':
        for _ in range(10):
            lines = lines.replace(op+" ", op)
            lines = lines.replace(" "+op, op)
    for op in ':?,':
        for _ in range(10):
            lines = lines.replace(op+"\n", op)
            lines = lines.replace("\n"+op, op)
    
    for _ in range(3):
        lines = lines.split("\n")
        ls = []
        for x in lines:
            if x.startswith("int") and ls[-1].startswith("int") and x[-1] != '{' and ls[-1][-1] != '{':
                ls[-1] = ls[-1][:-1] # drop semicolon
                ls.append(","+x[3:])
            else:
                ls.append(x)
        lines = "\n".join(ls)
        lines = lines.replace("\n,", ",")
    lines = lines.replace("}", "} ")
    lines = lines.replace("{", "{ ")
    lines = lines.replace(";", ";\n")
    lines = lines.replace(" \n", "\n")
    lines = lines.replace("return 0;", "")
    lines = lines.replace("\n\n", "\n")

    # Remove single-line statement braces
    lines = lines.split("\n")
    for i in range(len(lines)-3):
        if len(lines[i]) and lines[i][-1] == '{' and lines[i+2] == '}':
            lines[i] = lines[i][:-1]
            lines[i+2] = ""
    lines = "\n".join(lines)
    

#lines = re.sub("for\(i=0; i<(.*); i\+\+\){", "Z(\\1)", lines)

lines_no_quotes = re.sub(r'"[^"]*"', '', lines)
replace = collections.Counter(re.findall("[a-zA-Z_][a-zA-Z_0-9]*", lines_no_quotes))

for x in {'for', 'main', 'void', 'else', 'printf', 'if', 'include', 'sizeof', 'stdio', 'int', 'memcpy', 'string', 'while', 'return', 'atoi', 'getchar', 'memset', 'stdlib', 'long', 'define', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'malloc', 'unsigned', 'continue', 'fflush', 'stdout', 'Write', 'ROM', 'goto', 'Read', 'TEST', 'char', 'putchar','puts','calloc', 'fcntl', 'O_NONBLOCK', 'FILE', 'fgetc', 'fopen', 'exit'}:
    del replace[x]
del replace['Z']

i = 0

chrs = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPRSTUVWXYZ'
for x in replace:
    chrs = chrs.replace(x,"")

replace = sorted(replace.items(), key=lambda x: (-len(x[0]), -x[1]))

print("Can't shorten", replace[len(chrs):])

rep = {}
for x,count in list(replace)[:len(chrs)]:
    if len(x) > 1:
        #print("ZZ", x, chrs[i], count)
        rep[x] = chrs[i]
        lines = lines.replace(x, chrs[i])
        i += 1

"""
lines = lines.split("\n")
ls = []
for x in lines:
    if x.startswith("int") and '(' in x and ')' in x and x[-1] == '{':
        # function call
        ls.append(x.replace("int",""))
    else:
        ls.append(x)
lines = "\n".join(ls)
"""

if 'tmp' not in rep:
    rep['tmp'] = 'tmp'

lines = lines.split("\n")
lines = ["#define W for(i=0;i<tmp;i++)\n".replace("tmp", rep["tmp"]),
         "#define INC_Q *Q++".replace("INC_Q", rep["INC_Q"])
         ] + lines
lines = "\n".join(lines)
lines = lines.replace("\n ", "\n")
#lines = lines.replace("#include<stdlib.h>\n", "")
lines = lines.replace("( ", "(")
lines = lines.replace("! ", "!")
lines = lines.replace(", ", ",")
lines = lines.replace("\1", "%d")
lines = lines.replace("\2", ".h")
lines = lines.replace(")\n", ")")
lines = lines.replace("(void*)", "")
lines = lines.split("\n")
lines = lines[:3] + [cptr] + lines[3:]
lines = "\n".join(lines)
lines = lines.replace("\n\n", "\n")
lines = lines.replace(";\n;\n", ";\n")

lines = lines.replace("int ", "_ ")
lines = lines.replace("int*", "_*")
lines = "typedef int _;\n"+lines
lines = "#define Z while(\n" + lines.replace("while(", "Z ")
lines = lines.replace("char* Q", "char*Q")
lines = lines.replace("Z (", "Z(")
lines = lines.replace("Z !", "Z!")



print(len(lines))

open("/tmp/prog.c","w").write(lines)
