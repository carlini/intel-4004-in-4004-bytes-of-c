    ;;  R0 R1 R2 R3 have the first number
    ;;  R4 R5 R6 R7 have the second number

    nop                        
    nop                        
    ldm 0
    xch R0
    ldm 0
    xch R1
    ldm 0
    xch R2
    ldm 1
    xch R3

    ldm 0
    xch R4
    ldm 0
    xch R5
    ldm 0
    xch R6
    ldm 1
    xch R7
    
top:
    ld R3
    add R7
    daa
    xch R3

    ld R2
    add R6
    daa
    xch R2

    ld R1
    add R5
    daa
    xch R1

    ld R0
    add R4
    daa
    xch R0

    ldm 3
    xch R15

    ld R0
    xch R4
    xch R0

    ld R1
    xch R5
    xch R1

    ld R2
    xch R6
    xch R2

    ld R3
    xch R7
    xch R3


wait_sector_high:
    jcn     t wait_sector_high
wait_sector_low:
    jcn     tn  wait_sector_low
    
    ldm 0
    xch R13
    ldm 4
clear_print_head:   
    jms shift_in_number
    isz R13 clear_print_head
    
    ld R3
    jms maybe_print
    ld R2
    jms maybe_print
    ld R1
    jms maybe_print
    ld R0
    jms maybe_print
    

    ldm 4
    jms shift_in_number
    jms shift_in_number
    jms shift_in_number

    jms strike_print_head
    isz R15 wait_sector_high

    jms linefeed
    jun top

maybe_print:
    clc
    sub R15
    iac
    iac
    JCN NZ next_a
    ldm 6
    jun next_b
next_a:
    ldm 4
next_b:    
    jms shift_in_number
    bbl 0
    

shift_in_number:    
    wrr
    ldm 0
    wrr
    bbl 4

strike_print_head:
    ldm 2
    wmp

    ldm 0
    wmp
    bbl 0

linefeed:
    ldm 10
    wmp

    ldm 0
    wmp
    bbl 0
    
end:    
