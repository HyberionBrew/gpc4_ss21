# Register VM Instruction Set

## Instruction Format

__Op-Types depending on first two bits__:
- R-Type: `00`
- I-Type: `01`
- M-Type: `10`
- Other: `11`

### Register operation (R-Type)
```
OPCODE    R1                  R2                  RD
XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX
```

### Immediate operation (I-Type)
```
OPCODE    R1                  IM                                      RD
XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX  XXXXXXXX
```

### Load/Store (M-Type)
```
OPCODE    R1
XXXXXXXX  XXXXXXXX  XXXXXXXX
```

### Bytecode Format

- Indicate register byte length in preamble
- All registers are written to only once (YOWO)
- In and out streams with respective names and reg nums are written to the bytecode file header

## Instructions

### R-Type (MSB 00)
| opcode  | R-Instruction         |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 0x08    | Add                   | Add Integer Streams                                   |
| 0x09    | Mul                   | Multiply Integer Streams                              |
| 0x0A    | Sub                   | Subtract Integer Streams                              |
| 0x0B    | Div                   | Divide Integer Streams (round down)                   |
| 0x0C    | Mod                   | Calculate Modulo of Integer Streams                   |
| 0x0D    | Delay                 | Delay Operation (first operand int, second def)       |
| 0x0E    | Last                  | Last Operation (first operand int, second def)        |
| 0x0F    | TimeIn                | Time Operation on Integer Stream                      |
| 0x10    | TimeUn                | Time Operation on Unit Stream                         |
| 0x11    | MergeIn               | Merge Integer Streams                                 |
| 0x12    | MergeUn               | Merge Unit Streams                                    |
| 0x13    | Count                 | Count Operation                                       |

### I-Type (MSB 01)
| opcode  | I-Instruction         |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 0x48    | AddI                  | Add Immediate to Integer Stream                       |
| 0x49    | MulI                  | Multiply Immediate with Integer Stream                |
| 0x4A    | SubI                  | Subtract Immediate from Integer Stream                |
| 0x4B    | SubII                 | Subtract Integer Stream from Immediate                |
| 0x4C    | DivI                  | Divide Integer Stream by Immediate                    |
| 0x4D    | DivII                 | Divide Immediate by Integer Stream                    |
| 0x4E    | ModI                  | Calculate Modulo of Integer Stream by Immediate       |
| 0x4F    | ModII                 | Calculate Modulo of Immediate by Integer Stream       |
| 0x50    | Default               | Create Stream with int event of value IM at time 0    |

### M-Type (MSB 10)
| opcode  | M-Instruction         |   Semantics                                           | 
|---------|-----------------------|-------------------------------------------------------|
| 0x88    | Load                  | Load virtual register to VRAM immediately             |
| 0x89    | Load4                 | Load local Variable from ID (1 byte)                  |
| 0x8A    | Load6                 | Load global Variable from ID (1 byte)                 |
| 0x8B    | Load8                 | Store topmost stack value to local ID (1 byte)        |
| 0x8C    | Store                 | Download virtual register from VRAM (=> output stream)|
| 0x8D    | Free                  | Free pseudo register                                  |
| 0x8E    | Unit                  | Create Stream with def event at time 0                |

### Exit (0xFF)
| opcode  | Control Instruction   |   Semantics                                           |
|---------|-----------------------|-------------------------------------------------------|
| 0xFF    | Exit                  | Exit program (last instruction)                       |

