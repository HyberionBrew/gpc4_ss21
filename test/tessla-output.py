#!/bin/python3 -u

import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate output file from tessla spec and input")
    parser.add_argument('name', help="name of input and tessla spec file")
    parser.add_argument('--spec', dest="spec", help="name of tessla spec file (optional, if deviating from input file)")
    parser.add_argument('--output', dest="output", help="name of output file (optional, if deviating from input file)")
    parser.add_argument('--tessla-bin', dest="tessla_bin", default="/usr/bin/tessla-bin/tessla-assembly-1.2.2.jar", help="location of tessla binary")
    args = parser.parse_args()

    input=args.name
    trunc=input[:input.rindex(".")];
    tessla=trunc + ".tessla"
    output=trunc + ".out"
    tessla_bin=args.tessla_bin

    if args.spec is not None:
        tessla = args.spec
    if args.output is not None:
        output = args.output
    
    cmd_string = "java -jar {} interpreter {} {} > {}".format(tessla_bin, tessla, input, output)
    os.system(cmd_string)
    

if __name__ == "__main__":
    main()
