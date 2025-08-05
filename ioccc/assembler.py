#!/usr/bin/env python3
"""
Intel 4004 Assembler
Converts 4004 assembly language code to binary machine code
"""

import re
import sys
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

@dataclass
class Instruction:
    """Class for representing a 4004 instruction with its binary encoding"""
    mnemonic: str
    opcode: int
    length: int  # 1 or 2 bytes
    operand_type: str  # None, reg, regpair, data, address, condition


class Assembler:
    def __init__(self):
        self.labels = {}  # Symbol table for labels
        self.equs = {}    # Symbol table for EQU directives
        self.current_address = 0
        self.output_bytes = []
        self.errors = []
        self.init_instruction_set()

    def init_instruction_set(self):
        """Initialize the instruction set dictionary"""
        self.instructions = {
            # One-byte instructions
            "NOP": Instruction("NOP", 0x00, 1, None),
            "LDM": Instruction("LDM", 0xD0, 1, "data"),  # 0xD0 + data
            "LD": Instruction("LD", 0xA0, 1, "reg"),    # 0xA0 + reg
            "XCH": Instruction("XCH", 0xB0, 1, "reg"),   # 0xB0 + reg
            "ADD": Instruction("ADD", 0x80, 1, "reg"),   # 0x80 + reg
            "SUB": Instruction("SUB", 0x90, 1, "reg"),   # 0x90 + reg
            "INC": Instruction("INC", 0x60, 1, "reg"),   # 0x60 + reg
            "BBL": Instruction("BBL", 0xC0, 1, "data"),  # 0xC0 + data
            "JIN": Instruction("JIN", 0x31, 1, "regpair"),  # 0x30 + regpair + 1
            "SRC": Instruction("SRC", 0x21, 1, "regpair"),  # 0x20 + regpair + 1
            "FIN": Instruction("FIN", 0x30, 1, "regpair"),  # 0x30 + regpair + 0
            
            # I/O and RAM Instructions
            "RDM": Instruction("RDM", 0xE9, 1, None),
            "RD0": Instruction("RD0", 0xEC, 1, None),
            "RD1": Instruction("RD1", 0xED, 1, None),
            "RD2": Instruction("RD2", 0xEE, 1, None),
            "RD3": Instruction("RD3", 0xEF, 1, None),
            "RDR": Instruction("RDR", 0xEA, 1, None),
            "WRM": Instruction("WRM", 0xE0, 1, None),
            "WR0": Instruction("WR0", 0xE4, 1, None),
            "WR1": Instruction("WR1", 0xE5, 1, None),
            "WR2": Instruction("WR2", 0xE6, 1, None),
            "WR3": Instruction("WR3", 0xE7, 1, None),
            "WRR": Instruction("WRR", 0xE2, 1, None),
            "WMP": Instruction("WMP", 0xE1, 1, None),
            "ADM": Instruction("ADM", 0xEB, 1, None),
            "SBM": Instruction("SBM", 0xE8, 1, None),
            
            # Accumulator Group Instructions
            "CLB": Instruction("CLB", 0xF0, 1, None),
            "CLC": Instruction("CLC", 0xF1, 1, None),
            "IAC": Instruction("IAC", 0xF2, 1, None),
            "CMC": Instruction("CMC", 0xF3, 1, None),
            "CMA": Instruction("CMA", 0xF4, 1, None),
            "RAL": Instruction("RAL", 0xF5, 1, None),
            "RAR": Instruction("RAR", 0xF6, 1, None),
            "TCC": Instruction("TCC", 0xF7, 1, None),
            "DAC": Instruction("DAC", 0xF8, 1, None),
            "TCS": Instruction("TCS", 0xF9, 1, None),
            "STC": Instruction("STC", 0xFA, 1, None),
            "DAA": Instruction("DAA", 0xFB, 1, None),
            "KBP": Instruction("KBP", 0xFC, 1, None),
            "DCL": Instruction("DCL", 0xFD, 1, None),
            
            # Two-byte instructions
            "JUN": Instruction("JUN", 0x40, 2, "address"),  # 0x40 + high_addr, low_addr
            "JMS": Instruction("JMS", 0x50, 2, "address"),  # 0x50 + high_addr, low_addr
            "JCN": Instruction("JCN", 0x10, 2, "condition_address"),  # 0x10 + condition, address
            "ISZ": Instruction("ISZ", 0x70, 2, "reg_address"),  # 0x70 + reg, address
            "FIM": Instruction("FIM", 0x20, 2, "regpair_data"),  # 0x20 + regpair, data
        }
        
        # Add P0-P7 as aliases for register pairs
        self.register_pairs = {f"P{i}": i for i in range(8)}
        self.register_pairs.update({f"{i}<": i for i in range(8)})
        self.register_pairs.update({f"R{i}": i for i in range(8)})
        self.register_pairs.update({f"r{i}": i for i in range(8)})
        
        # Add R0-R15 as aliases for registers
        self.registers = {f"R{i}": i for i in range(16)}
        
        # Add condition codes
        self.conditions = {
            "T": 0x1,     # Jump if carry = 1
            "NT": 0x9,    # Jump if no carry
            "C": 0x2,     # Jump if zero
            "NC": 0xA,    # Jump if not zero
            "Z": 0x4,     # Jump if test = 0
            "NZ": 0xC,    # Jump if test = 1
            
        }
        self.conditions = {tuple(sorted(k)): v for k, v in self.conditions.items()}
        #print(self.conditions)

    def parse_line(self, line: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Parse a line of assembly code into label, instruction, operand, and comment"""
        # Remove comment if present
        comment_match = re.search(r';(.*)', line)
        comment = comment_match.group(1).strip() if comment_match else None
        line = re.sub(r';.*', '', line).strip()
        
        if not line:
            return None, None, None, comment
        
        # Check for label
        label_match = re.match(r'([A-Za-z0-9_]+):', line)
        label = None
        if label_match:
            label = label_match.group(1)
            line = line[label_match.end():].strip()
        
        # Check for EQU directive
        equ_match = re.match(r'([A-Za-z0-9_]+)\s+EQU\s+(.*)', line)
        if equ_match:
            symbol = equ_match.group(1)
            value_str = equ_match.group(2).strip()
            
            # Parse the EQU value
            try:
                if value_str.startswith('0x'):
                    value = int(value_str[2:], 16)
                elif value_str.startswith('0b'):
                    value = int(value_str[2:], 2)
                else:
                    value = int(value_str)
                
                self.equs[symbol] = value
            except ValueError:
                self.errors.append(f"Invalid EQU value: {value_str}")
            
            return None, "EQU", value_str, comment
        
        # Extract instruction and operand
        parts = line.split(None, 1)
        instruction = parts[0].upper() if parts else None
        operand = parts[1].strip() if len(parts) > 1 else None
        
        return label, instruction, operand, comment

    def substitute_equ_symbols(self, operand_str):
        """Replace any EQU symbols in the operand string with their values"""
        # First check if the entire operand is an EQU symbol
        if operand_str in self.equs:
            return str(self.equs[operand_str])
        
        # Otherwise, look for EQU symbols within the operand
        # This is useful for expressions like "LABEL+OFFSET"
        for symbol, value in self.equs.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(symbol) + r'\b'
            operand_str = re.sub(pattern, str(value), operand_str)
            
        return operand_str
        
    def parse_operand(self, operand_str: str, operand_type: str) -> Union[int, Tuple[int, int]]:
        """Parse operand string based on its type"""
        if operand_type is None:
            return None
            
        # Substitute any EQU symbols in the operand
        operand_str = self.substitute_equ_symbols(operand_str)
            
        # Handle different operand types
        if operand_type == "reg":
            if operand_str in self.registers:
                return self.registers[operand_str]
            try:
                reg_num = int(operand_str)
                if 0 <= reg_num <= 15:
                    return reg_num
                else:
                    self.errors.append(f"Register number {reg_num} out of range (0-15)")
            except ValueError:
                self.errors.append(f"Invalid register: {operand_str}")
                
        elif operand_type == "regpair":
            if operand_str in self.register_pairs:
                return self.register_pairs[operand_str]
            try:
                pair_num = int(operand_str)
                if 0 <= pair_num <= 7:
                    return pair_num
                else:
                    self.errors.append(f"Register pair number {pair_num} out of range (0-7)")
            except ValueError:
                self.errors.append(f"Invalid register pair: {operand_str}")
                
        elif operand_type == "data":
            try:
                # Handle different number formats (decimal, hex, binary)
                if operand_str.startswith('0x'):
                    value = int(operand_str[2:], 16)
                elif operand_str.startswith('0b'):
                    value = int(operand_str[2:], 2)
                else:
                    value = int(operand_str)
                    
                if 0 <= value <= 15:
                    return value
                else:
                    self.errors.append(f"Data value {value} out of range (0-15)")
            except ValueError:
                self.errors.append(f"Invalid data value: {operand_str}")
                
        elif operand_type == "address":
            try:
                # Handle label or direct address
                if operand_str in self.labels:
                    address = self.labels[operand_str]
                elif operand_str.startswith('0x'):
                    address = int(operand_str[2:], 16)
                else:
                    address = int(operand_str)
                    
                if 0 <= address <= 0xFFF:  # 12-bit address
                    high_addr = (address >> 8) & 0xF
                    low_addr = address & 0xFF
                    return high_addr, low_addr
                else:
                    self.errors.append(f"Address {address} out of range (0-0xFFF)")
            except ValueError:
                # If we can't resolve the label now, we'll handle it in the second pass
                return operand_str
                
        elif operand_type == "condition_address":
            # Parse condition and address for JCN
            parts = operand_str.split(' ', 1)
            if len(parts) != 2:
                self.errors.append(f"Invalid operand format for condition+address: {operand_str}")
                return None
                
            condition_str, address_str = parts
            condition_str = condition_str.strip().upper()
            address_str = address_str.strip()
            
            # Handle condition (substitute EQU if needed)
            condition_str = self.substitute_equ_symbols(condition_str)
            if tuple(sorted(condition_str)) in self.conditions:
                condition = self.conditions[tuple(sorted(condition_str))]
            else:
                try:
                    condition = int(condition_str, 0)
                    if not (0 <= condition <= 15):
                        self.errors.append(f"Condition {condition} out of range (0-15)")
                except ValueError:
                    self.errors.append(f"Invalid condition: {condition_str}")
                    return None
            
            # Handle address (substitute EQU if needed)
            address_str = self.substitute_equ_symbols(address_str)
            try:
                if address_str in self.labels:
                    address = self.labels[address_str]
                elif address_str.startswith('0x'):
                    address = int(address_str[2:], 16)
                else:
                    address = int(address_str)
                    
                if 0 <= address <= 0xFF:  # 8-bit address for JCN
                    return condition, address
                else:
                    self.errors.append(f"Address {address} out of range (0-0xFF) for JCN")
            except ValueError:
                # If we can't resolve the label now, we'll handle it in the second pass
                return condition, address_str
                
        elif operand_type == "reg_address":
            # Parse register and address for ISZ
            parts = operand_str.split(' ', 1)
            if len(parts) != 2:
                self.errors.append(f"Invalid operand format for reg+address: {operand_str}")
                return None
                
            reg_str, address_str = parts
            reg_str = reg_str.strip().upper()
            address_str = address_str.strip()
            
            # Handle register (substitute EQU if needed)
            reg_str = self.substitute_equ_symbols(reg_str)
            if reg_str in self.registers:
                reg = self.registers[reg_str]
            else:
                try:
                    reg = int(reg_str)
                    if not (0 <= reg <= 15):
                        self.errors.append(f"Register {reg} out of range (0-15)")
                except ValueError:
                    self.errors.append(f"Invalid register: {reg_str}")
                    return None
            
            # Handle address (substitute EQU if needed)
            address_str = self.substitute_equ_symbols(address_str)
            try:
                if address_str in self.labels:
                    address = self.labels[address_str]
                elif address_str.startswith('0x'):
                    address = int(address_str[2:], 16)
                else:
                    address = int(address_str)
                    
                if 0 <= address <= 0xFF:  # 8-bit address for ISZ
                    return reg, address
                else:
                    self.errors.append(f"Address {address} out of range (0-0xFF) for ISZ")
            except ValueError:
                # If we can't resolve the label now, we'll handle it in the second pass
                return reg, address_str
                
        elif operand_type == "regpair_data":
            # Parse register pair and data for FIM
            parts = operand_str.split(' ', 1)
            if len(parts) != 2:
                self.errors.append(f"Invalid operand format for regpair+data: {operand_str}")
                return None
                
            regpair_str, data_str = parts
            regpair_str = regpair_str.strip().upper()
            data_str = data_str.strip()
            
            # Handle register pair (substitute EQU if needed)
            regpair_str = self.substitute_equ_symbols(regpair_str)
            if regpair_str in self.register_pairs:
                regpair = self.register_pairs[regpair_str]
            else:
                try:
                    regpair = int(regpair_str)
                    if not (0 <= regpair <= 7):
                        self.errors.append(f"Register pair {regpair} out of range (0-7)")
                except ValueError:
                    self.errors.append(f"Invalid register pair: {regpair_str}")
                    return None
            
            # Handle data (substitute EQU if needed)
            data_str = self.substitute_equ_symbols(data_str)
            try:
                if data_str.startswith('0x'):
                    data = int(data_str[2:], 16)
                elif data_str.startswith('0b'):
                    data = int(data_str[2:], 2)
                else:
                    data = int(data_str)
                    
                if 0 <= data <= 0xFF:  # 8-bit data for FIM
                    return regpair, data
                else:
                    self.errors.append(f"Data value {data} out of range (0-0xFF) for FIM")
            except ValueError:
                self.errors.append(f"Invalid data value: {data_str}")
        
        return None

    def resolve_label_address(self, label_str, instruction, line_number):
        """Resolve a label string to its address in the second pass"""
        if label_str in self.labels:
            return self.labels[label_str]
        
        self.errors.append(f"Line {line_number}: Undefined label '{label_str}' for instruction {instruction}")
        return 0  # Return a default value to avoid crashes

    def assemble(self, source_lines: List[str]) -> bytes:
        """
        Assemble the source code into machine code
        
        This is a two-pass assembler:
        1. First pass: collect labels and their addresses, process EQU directives
        2. Second pass: generate machine code
        """
        # First pass: collect labels and process EQU directives
        self.current_address = 0
        self.labels = {}
        self.equs = {}
        
        for line_number, line in enumerate(source_lines, 1):
            label, instruction, operand, _ = self.parse_line(line)
            
            if label:
                if label in self.labels:
                    self.errors.append(f"Line {line_number}: Duplicate label '{label}'")
                else:
                    self.labels[label] = self.current_address
            
            if instruction == "EQU":
                # EQU directive is already processed in parse_line
                continue
            elif instruction:
                if instruction in self.instructions:
                    self.current_address += self.instructions[instruction].length
                elif instruction.startswith('0X'):
                    self.current_address += 1
                else:
                    self.errors.append(f"Line {line_number}: Unknown instruction '{instruction}'")
        
        # Second pass: generate machine code
        self.current_address = 0
        self.output_bytes = []
        
        for line_number, line in enumerate(source_lines, 1):
            _, instruction, operand, _ = self.parse_line(line)
            
            if not instruction or instruction == "EQU":
                continue

            
            if instruction.upper().startswith('0X'):
                # Handle raw hex byte
                try:
                    raw_byte = int(instruction, 16)
                    if 0 <= raw_byte <= 0xFF:
                        self.output_bytes.append(raw_byte)
                        self.current_address += 1
                    else:
                        self.errors.append(f"Line {line_number}: Hex value {instruction} out of range (0-0xFF)")
                except ValueError:
                    self.errors.append(f"Line {line_number}: Invalid hex value '{instruction}'")
                continue
            elif instruction not in self.instructions:
                continue  # Already reported in first pass
            else:
                instr = self.instructions[instruction]
            
            if instr.operand_type is None:
                # Instructions with no operand
                self.output_bytes.append(instr.opcode)
            elif instr.operand_type == "reg":
                # Register operand
                reg = self.parse_operand(operand, "reg")
                if reg is not None:
                    self.output_bytes.append(instr.opcode | reg)
            elif instr.operand_type == "regpair":
                # Register pair operand (for SRC, JIN, FIN)
                regpair = self.parse_operand(operand, "regpair")
                if regpair is not None:
                    # For SRC and JIN, we OR with regpair*2+1
                    # For FIN, we OR with regpair*2
                    if instruction in ["SRC", "JIN"]:
                        self.output_bytes.append(instr.opcode | (regpair << 1))
                    else:  # FIN
                        self.output_bytes.append(instr.opcode | (regpair << 1))
            elif instr.operand_type == "data":
                # Data operand (for LDM, BBL)
                data = self.parse_operand(operand, "data")
                if data is not None:
                    self.output_bytes.append(instr.opcode | data)
            elif instr.operand_type == "address":
                # Address operand (for JUN, JMS)
                addr = self.parse_operand(operand, "address")
                if isinstance(addr, str):
                    # Try to resolve label in second pass
                    address = self.resolve_label_address(addr, instruction, line_number)
                    high_addr = (address >> 8) & 0xF
                    low_addr = address & 0xFF
                    self.output_bytes.append(instr.opcode | high_addr)
                    self.output_bytes.append(low_addr)
                elif addr is not None:
                    high_addr, low_addr = addr
                    self.output_bytes.append(instr.opcode | high_addr)
                    self.output_bytes.append(low_addr)
            elif instr.operand_type == "condition_address":
                # Condition and address operand (for JCN)
                parsed = self.parse_operand(operand, "condition_address")
                if parsed is not None:
                    condition, address = parsed
                    if isinstance(address, str):
                        # Try to resolve label in second pass
                        addr_value = self.resolve_label_address(address, instruction, line_number)
                        if addr_value > 0xFF:
                            self.errors.append(f"Line {line_number}: Address for label '{address}' out of range for JCN")
                            addr_value &= 0xFF  # Truncate to 8 bits to avoid errors
                        self.output_bytes.append(instr.opcode | condition)
                        self.output_bytes.append(addr_value & 0xFF)
                    else:
                        self.output_bytes.append(instr.opcode | condition)
                        self.output_bytes.append(address)
            elif instr.operand_type == "reg_address":
                # Register and address operand (for ISZ)
                parsed = self.parse_operand(operand, "reg_address")
                if parsed is not None:
                    reg, address = parsed
                    if isinstance(address, str):
                        # Try to resolve label in second pass
                        addr_value = self.resolve_label_address(address, instruction, line_number)
                        if addr_value > 0xFF:
                            self.errors.append(f"Line {line_number}: Address for label '{address}' out of range for ISZ")
                            addr_value &= 0xFF  # Truncate to 8 bits to avoid errors
                        self.output_bytes.append(instr.opcode | reg)
                        self.output_bytes.append(addr_value & 0xFF)
                    else:
                        self.output_bytes.append(instr.opcode | reg)
                        self.output_bytes.append(address)
            elif instr.operand_type == "regpair_data":
                # Register pair and data operand (for FIM)
                parsed = self.parse_operand(operand, "regpair_data")
                if parsed is not None:
                    regpair, data = parsed
                    self.output_bytes.append(instr.opcode | (regpair << 1))
                    self.output_bytes.append(data)
            else:
                raise
            
            self.current_address += instr.length
        
        return bytes(self.output_bytes)

def main():
    parser = argparse.ArgumentParser(description='Intel 4004 Assembler')
    parser.add_argument('input_file', help='input assembly file')
    parser.add_argument('-o', '--output', help='output binary file')
    parser.add_argument('-l', '--listing', help='output listing file')
    args = parser.parse_args()
    
    # Read input file
    with open(args.input_file, 'r') as f:
        source_lines = f.readlines()
    
    assembler = Assembler()
    binary = assembler.assemble(source_lines)
    
    if assembler.errors:
        for error in assembler.errors:
            print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    
    # Write binary output
    if args.output:
        with open(args.output, 'wb') as f:
            f.write(binary)
    else:
        # Print as hex
        hex_output = ' '.join(f'{b:02X}' for b in binary)
        print(hex_output)
    
    # Generate listing if requested
    if args.listing:
        with open(args.listing, 'w') as f:
            address = 0
            for i, line in enumerate(source_lines):
                label, instruction, operand, comment = assembler.parse_line(line)
                
                if instruction == "EQU":
                    # Add EQU directive to listing without address or machine code
                    f.write(f'     :          {line.rstrip()}\n')
                elif instruction and instruction in assembler.instructions:
                    instr = assembler.instructions[instruction]
                    if instr.length == 1:
                        byte_str = f'{binary[address]:02X}'
                        f.write(f'{address:04X}: {byte_str}       {line.rstrip()}\n')
                        address += 1
                    else:  # 2-byte instruction
                        byte_str = f'{binary[address]:02X} {binary[address+1]:02X}'
                        f.write(f'{address:04X}: {byte_str}    {line.rstrip()}\n')
                        address += 2
                else:
                    # Labels or comments only
                    f.write(f'     :          {line.rstrip()}\n')

if __name__ == '__main__':
    main()
