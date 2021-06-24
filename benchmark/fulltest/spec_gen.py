#!/bin/usr/python3 -u

import argparse
import copy
import os
import string
import random
from input_gen import generate_output

# global integer limits
INT_MIN = 1
INT_MAX = 100

# format strings for statements and expressions
IN_FORMAT = "in {}: Events[{}]"
OUT_FORMAT = "out {}"
DEF_FORMAT = "def {} = {}"
ARITH_FORMAT = "{} {} {}"

ALPHABET = string.ascii_lowercase
string_index = 1

# stream "enums"
INT_STREAM = 0
UNIT_STREAM = 1
STREAM = 2

PICK_CHANCE = 0.75

ARITH_OPS = ["+", "*"]  # exclude "-", "/", "%" for now to not kill delay :)

RESERVED_KW = {"delay", "last", "time", "merge", "count", "in", "out", "def", "next", "as"}

SEED_SIZE = 5


def string_gen(x):
    digs = ALPHABET
    base = len(ALPHABET)
    digits = []
    while x:
        digits.append(digs[x % base - 1])
        x = int(x / base)
    digits.reverse()
    return ''.join(digits)


class Operation:
    def __init__(self, name, types, ret_type):
        self.name = name
        self.types = types
        self.ret_type = ret_type

    def get_stat(self, ops):
        assert (len(ops) == len(self.types))
        op_list = ", ".join(ops)
        return "{}({})".format(self.name, op_list)

    def __repr__(self):
        return self.name + "()"


class UnArithOp:
    def __init__(self):
        self.types = [INT_STREAM]
        self.ret_type = INT_STREAM

    def get_stat(self, ops):
        assert (len(ops) == len(self.types))
        op_int = random.randint(INT_MIN, INT_MAX)
        operator = random.choice(ARITH_OPS)
        if random.random() <= 0.5:
            return ARITH_FORMAT.format(op_int, operator, ops[0])
        else:
            return ARITH_FORMAT.format(ops[0], operator, op_int)
        pass

    def __repr__(self):
        return "unArithOp()"


class BinArithOp:
    def __init__(self):
        self.types = [INT_STREAM, INT_STREAM]
        self.ret_type = INT_STREAM

    def get_stat(self, ops):
        assert (len(ops) == len(self.types))
        operator = random.choice(ARITH_OPS)
        return ARITH_FORMAT.format(ops[0], operator, ops[1])

    def __repr__(self):
        return "binArithOp()"


class Stream:
    def __init__(self, stream_type, is_input):
        self.name = generate_name()
        self.is_input = is_input
        self.stream_type = stream_type

    def type_str(self):
        if self.stream_type == INT_STREAM:
            return "Int"
        elif self.stream_type == UNIT_STREAM:
            return "Unit"
        else:
            raise RuntimeError("Found invalid stream type")

    def __repr__(self):
        type_str = "UNDEFINED"
        if self.stream_type == INT_STREAM:
            type_str = "Int"
        elif self.stream_type == UNIT_STREAM:
            type_str = "Unit"
        in_str = "Input" if self.is_input else "Intermediate"
        return "({}: {} [{}])".format(self.name, type_str, in_str)


OPERATIONS = [
    # Operation("delay", [INT_STREAM, STREAM], UNIT_STREAM),
    Operation("last", [INT_STREAM, STREAM], INT_STREAM),
    Operation("time", [STREAM], INT_STREAM),
    Operation("merge", [INT_STREAM, INT_STREAM], INT_STREAM),
    Operation("merge", [UNIT_STREAM, UNIT_STREAM], UNIT_STREAM),
    Operation("count", [STREAM], INT_STREAM),
    UnArithOp(),
    BinArithOp(),
]


def generate_name():
    global string_index
    success = False
    s = ""
    while not success:
        s = string_gen(string_index)
        string_index += 1
        success = s not in RESERVED_KW
    return s


"""
idea: make list of functions (?) and pick length elements. Then, lazily iterate through the list and get
as many streams as required, prefer to use last ones for propagation purposes (maybe non-deterministically /
decreased chance for elements lower in the "stack"?
"""


# gets _names_ of streams with fitting type in stream stack and possibly allocates new ones
def get_streams(stream_stack, stream_types):
    not_found = copy.deepcopy(stream_types)
    not_found.reverse()
    op_streams = []
    for s in stream_stack[::-1]:
        if s.stream_type == not_found[-1] or not_found[-1] == STREAM:
            if random.random() < PICK_CHANCE:
                op_streams.append(s.name)
                not_found.pop()
        if len(not_found) == 0:
            break
    for t in not_found:
        # if operands have not been found, alloc new input (!) streams
        op_streams.append(alloc_new(stream_stack, t, True))
    return op_streams


# alloc new stream in stream stack and return name of allocated stream
def alloc_new(stream_stack, stream_type, is_input):
    if stream_type == STREAM:
        stream_type = INT_STREAM if random.random() <= 0.5 else UNIT_STREAM
    new_stream = Stream(stream_type, is_input)
    stream_stack.append(new_stream)
    return new_stream.name


def generate_spec(length, seed):
    random.seed(seed)

    op_list = []
    def_list = []
    stream_stack = []

    # generate list of length elements
    for _ in range(length):
        op_list.append(random.choice(OPERATIONS))

    for op in op_list:
        operands = get_streams(stream_stack, op.types)
        new_name = alloc_new(stream_stack, op.ret_type, False)
        def_list.append(DEF_FORMAT.format(new_name, op.get_stat(operands)))

    # generate input stream statements
    in_list = []
    out_list = []
    int_names = []
    unit_names = []
    for s in stream_stack:
        if s.is_input:
            in_list.append(IN_FORMAT.format(s.name, s.type_str()))
            if s.stream_type == INT_STREAM:
                int_names.append(s.name)
            elif s.stream_type == UNIT_STREAM:
                unit_names.append(s.name)
            else:
                raise RuntimeError("Found stream with undefined type")
        else:
            out_list.append(OUT_FORMAT.format(s.name, s.type_str()))

    stat_list = []
    stat_list.extend(in_list)
    stat_list.extend(def_list)
    stat_list.extend(out_list)
    return int_names, unit_names, "\n".join(stat_list)


def random_string(str_size):
    return "".join(random.choice(string.ascii_letters) for _ in range(str_size))


def generate_test(spec_length, trace_length, seed, target_folder):

    # generate specification
    int_names, unit_names, spec = generate_spec(spec_length, seed)

    # generate int_weights and unit weights
    random.seed(seed)
    int_weights = []
    unit_weights = []
    for _ in int_names:
        int_weights.append(random.random())
    for _ in unit_names:
        unit_weights.append(random.random())

    # generate input trace
    trace = generate_output(int_names, unit_names, int_weights, unit_weights, INT_MIN, INT_MAX, trace_length, seed)

    # write to files
    test_name = "test_{}_sl{}_tl{}".format(seed, spec_length, trace_length)
    spec_name = test_name + ".tessla"
    trace_name = test_name + ".in"
    with open(os.path.join(target_folder, spec_name), "w") as file:
        file.write(spec)
    with open(os.path.join(target_folder, trace_name), "w") as file:
        file.write(trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-sl", "--spec-length", help="Specifies number of operation that should be present in the "
                                                     "spec file",
                        type=int, default=42)
    parser.add_argument("-n", "--num-tests", help="Specifies number specifications that should be generated",
                        type=int, default=1)
    parser.add_argument("-s", "--seed", help="Initial seed for specifications and input file", type=str)
    parser.add_argument("-tl", "--trace-length", help="Specifies maximum timestamp for input (=trace) file streams",
                        type=int, default=100)
    parser.add_argument("-o", "--output-folder", help="The folder to write the specification and input trace to",
                        type=str, default="")

    args = parser.parse_args()

    # set seed
    seed = random_string(SEED_SIZE) if args.seed is None else args.seed

    out_folder = args.output_folder

    # create output folder if it does not exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for _ in range(args.num_tests):
        generate_test(args.spec_length, args.trace_length, seed, out_folder)
        # get next seed
        random.seed(seed)
        seed = random_string(SEED_SIZE)


if __name__ == "__main__":
    main()
