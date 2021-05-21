# Register VM Instruction Set

## Instruction Format
### Register operation (R-Type)
OPCODE    R1                  R2                  RD
XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX

### Immediate operation (I-Type)
OPCODE    R1                  IM                                      RD
XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX

### Load/Store (M-Type)
OPCODE    R1
XXXXXXXX  XXXXXXXX  XXXXXXXX

### Bytecode Format

- All registers are written to only once (YOWO)
- In and out streams with respective names and reg nums are written to the bytecode file header

## Instructions

- **TODO ADD SEMANTICS**

| opcode  | R-Instruction         |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 8       | Add                   | Add Integer Values                                    |
| 9       | Mul                   | Subtract Integer Values                               |
| 10      | Sub                   | Multiply Integer Values                               |
| 11      | Div                   | Divide Integer Values (Result is truncated Integer)   |
| 12      | Mod                   | Calculate Modulo of Integers                          |
| 13      | Delay                 | Calculate Modulo of Integers                          |
| 15      | Last                  | Calculate Modulo of Integers                          |
| 17      | Time                  | Calculate Modulo of Integers                          |
| 18      | MergeIn               | Calculate Modulo of Integers                          |
| 19      | MergeUn               | Calculate Modulo of Integers                          |
| 20      | Count                 | Calculate Modulo of Integers                          |

| opcode  | I-Instruction         |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 30      | AddI                  | Compare Integers (Greater)                            |
| 31      | MulI                  | Compare Integers (Less than)                          |
| 32      | SubI                  | Compare Integers (Greater Equals)                     |
| 33      | SubII                 | Compare Integers (Greater Equals)                     |
| 34      | DivI                  | Compare Integers (Less than or Equals)                |
| 35      | DivII                 | Compare Integers (Less than or Equals)                |
| 36      | ModI                  | Negate Integer (invert Sign)                          |
| 37      | ModII                 | Negate Integer (invert Sign)                          |
| 38      | Default               | Create Stream with int event of value IM at time 0    |

| opcode  | M-Instruction         |   Semantics                                           | 
|---------|-----------------------|-------------------------------------------------------|
| 40      | Load                  | Load virtual register to VRAM immediately             |
| 44      | Load4                 | Load local Variable from ID (1 byte)                  |
| 46      | Load6                 | Load global Variable from ID (1 byte)                 |
| 48      | Load8                 | Store topmost stack value to local ID (1 byte)        |
| 55      | Store                 | Download virtual register from VRAM (=> output stream)|
| 56      | Free                  | Free pseudo register                                  |
| 57      | Unit                  | Create Stream with unit event at time 0               |

| opcode  | Control Instruction   |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 255     | Exit                  | Exit program (last instruction)                       |

