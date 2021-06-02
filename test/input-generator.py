#!/usr/bin/env python3 -u

import argparse
import random
import string
import sys

UNITSTREAM_NAMES = list(string.ascii_lowercase)
INTSTREAM_NAMES = list(string.ascii_lowercase)
INTSTREAM_NAMES.reverse()


def main():
    parser = argparse.ArgumentParser(description="Generate random input for tessla")
    parser.add_argument("--length", dest="length", type=int, default=2500, help="maximum timestamp on all streams")
    parser.add_argument("--unitweights", dest="unitweights", type=float, nargs="*",
                        help="weights associated with unit streams")
    parser.add_argument("--intweights", dest="intweights", type=float, nargs="*",
                        help="weights associated with int streams")
    parser.add_argument("--intmax", dest="intmax", type=int, default=50,
                        help="maximum of range for outputs on int streams")
    parser.add_argument("--output", dest="output", type=argparse.FileType("w"), nargs="?", default=sys.stdout,
                        help="output file name")

    args = parser.parse_args()

    unitstreams = list(zip(UNITSTREAM_NAMES, args.unitweights))
    intstreams = list(zip(INTSTREAM_NAMES, args.intweights))

    for i in range(0, args.length + 1):
        for (name, weight) in unitstreams:
            if weight <= random.random():
                args.output.write("{}: {} = ()\n".format(i, name))
        for (name, weight) in intstreams:
            if weight <= random.random():
                args.output.write("{}: {} = {}\n".format(i, name, random.randint(0, args.intmax)))


if __name__ == "__main__":
    main()
