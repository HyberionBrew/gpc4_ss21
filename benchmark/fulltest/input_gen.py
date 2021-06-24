#!/usr/bin/env python3 -u

import argparse
import random
import string
import sys

UNITSTREAM_NAMES = list(string.ascii_lowercase)
INTSTREAM_NAMES = list(string.ascii_lowercase)
INTSTREAM_NAMES.reverse()


def main():
    parser = argparse.ArgumentParser(description="Generate random input for TeSSLa")
    parser.add_argument("--length", dest="length", type=int, default=2500, help="maximum timestamp on all streams")
    parser.add_argument("-uw", "--unitweights", dest="unitweights", type=float, nargs="*",
                        help="weights associated with unit streams")
    parser.add_argument("-iw", "--intweights", dest="intweights", type=float, nargs="*",
                        help="weights associated with int streams")
    parser.add_argument("-iM", "--intmax", dest="intmax", type=int, default=50,
                        help="maximum of range for outputs on int streams")
    parser.add_argument("-im" "--intmin", dest="intmin", type=int, default=1,
                        help="minimum of range for outputs on int streams")
    parser.add_argument("-o", "--output", dest="output", type=str, nargs="?", default=sys.stdout,
                        help="output file name")

    args = parser.parse_args()

    out = generate_output(INTSTREAM_NAMES, UNITSTREAM_NAMES, args.intweights, args.unitweights,
                          args.intmin, args.intmax, args.length)

    with open(args.output, "w") as outfile:
        outfile.write(out)


def generate_output(int_names, unit_names, int_weights, unit_weights, int_min, int_max, length, seed):
    random.seed(seed)
    if unit_weights is None:
        unit_streams = []
    else:
        unit_streams = list(zip(unit_names, unit_weights))
    if int_weights is None:
        int_streams = []
    else:
        int_streams = list(zip(int_names, int_weights))

    output = []
    for i in range(1, length + 1):
        for (name, weight) in unit_streams:
            if weight <= random.random():
                output.append("{}: {} = ()".format(i, name))
        for (name, weight) in int_streams:
            if weight <= random.random():
                output.append("{}: {} = {}".format(i, name, random.randint(int_min, int_max)))
    return "\n".join(output)


if __name__ == "__main__":
    main()
