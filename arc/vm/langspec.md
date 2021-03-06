# Flat TeSSLa Specification

## Flat TeSSLa Language

In our (reduced) version of TeSSLa, we consider a hybrid version of flat TeSSLa(f) specifications
and the official V1.0 TeSSLa specification.

## Syntax

### General
- A program consists of multiple, new-line terminated statements
- Python-style (single-line) comments (start with `#`)

### Variables

#### Types

There are two types of streams in our TeSSLa subset: `Event[Int]` and `Event[Unit]`.
These mark streams _exclusively_ containing Integer or Unit Values, respectively.


### Comments
**Python-style (single line)**
```
# COMMENT TEXT 

# This is a comment 

def x = () # This is also a comment
```
## Grammar

All enquoted strings are terminals.
Unenquoted `(`, `)`, `*`, `[`, `]`, `?`,`-` and `+` have the usual regular expression semantics.

```
Program :   ( Stat '\n' )*

Stat    :   Define
        |   Assign 
        |   Input 
        |   Output
        
Input   :   'in' ID ':' TYPE '\n'
        
Output  :   'out' ID '\n'

Define  :   'def' ID '=' RHS '\n'

RHS     :   ID
        |   UnFunc
        |   BinFunc
        |   SLift
        |   Arith
        |   Unit
        |   Default

UnFunc  :   Time
        |   Count

BinFunc :   Delay
        |   Last
        |   Merge
 
Unit    :   '()'

Default :   ICONST

Delay   :   'delay' '(' ID ',' ID ')'

Last    :   'last' '(' ID ',' ID ')'

Time    :   'time' '(' ID ')'

Merge   :   'merge' '(' ID ',' ID ')'

Count   :   'count' '(' ID ')'

SLift   :   ID '+' ID
        |   ID '*' ID
        |   ID '-' ID
        |   ID '/' ID
        |   ID '%' ID

Arith   :   ID '+' ICONST
        |   ID '*' ICONST
        |   ID '-' ICONST
        |   ID '/' ICONST
        |   ID '%' ICONST
        |   ICONST '+' ID
        |   ICONST '*' ID
        |   ICONST '-' ID
        |   ICONST '/' ID
        |   ICONST '%' ID
        
ID      :   [A-Za-z][0-9A-Za-z_]*

ICONST  :   -?[0-9]+

TYPE    :   'Events[Int]'
        |   'Events[Unit]'
```


## Keywords

| Keyword       | Used for                                 |
|---------------|------------------------------------------|
| `def`      | Variable assignment |
| `in`,`out`      | Marking as Input / Output |
| `delay`, `last`, `time`      | Basic Operators |
| `merge`,`count`      | Compound Operators |
| `next`     | Reserved |

