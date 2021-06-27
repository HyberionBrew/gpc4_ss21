#!/bin/usr/python3 -u

import argparse
import copy
import math
import os
import string
import random
from input_gen import generate_property_trace

# global integer limits
INT_MIN = 1
INT_MAX = 100

# format strings for statements and expressions
IN_FORMAT = "in {}: Events[{}]"
OUT_FORMAT = "out {}"
DEF_FORMAT = "def {} = {}"
ARITH_FORMAT = "{} {} {}"

# random string generator state and alphabet
ALPHABET = string.ascii_lowercase
string_index = 1

# stream "enums"
INT_STREAM = "INT_STREAM"
UNIT_STREAM = "UNIT_STREAM"
STREAM = "STREAM"

# stream operations
DELAY = "delay"
LAST = "last"
TIME = "time"
MERGE = "merge"
COUNT = "count"

PICK_CHANCE = 0.75

ARITH_OPS = ["+", "-", "/", "%"]

RESERVED_KW = {DELAY, LAST, TIME, MERGE, COUNT, "in", "out", "def", "as", "if", "type", "module", "then", "else"}

SEED_SIZE = 5

MODE_DEBUG = "debug"
MODE_BENCHMARK = "benchmark"

POSITIVE = "POSITIVE"
NON_POSITIVE = "NON_POSITIVE"
UNIT = "UNIT"
DONT_CARE = "DONT_CARE"

# don't care requirement for better readability
DONT_CARE_REQ = (DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE)

MAX_INTEGER = 2147483647

"""
REQUIREMENTS:
requirements and stream properties are a 5-tuple of the following:
(IS_POSITIVE, MAX_VAL, ZERO_TS, MAX_SIZE, MAX_TIMESTAMP)
-   IS_POSITIVE: marks whether a integer stream is positive (POSITIVE) or non-positive (NON_POSITIVE). Unit streams
    are marked with UNIT
-   MAX_VAL: integer denoting the theoretical max value of a stream
-   ZERO_TS: boolean denoting whether a stream has an event at timestamp 0
-   MAX_SIZE: integer denoting the theoretical max size of a stream
-   MAX_TIMESTAMP: integer denoting the max timestamp of a stream

requirements may specify DONT_CARE in a field where the value is not important. In cases where a value is not important
or undefined, this may also be used in stream properties.

DONT_CARE in stream properties is only allowed in the following combinations (_ marks values that may be arbitrary in 
the specified value range):
-   (NON_POSITIVE, DONT_CARE, _, _, _)
-   (UNIT, DONT_CARE, _, _, _)

The requirements system works as follows:
1) An operation is "fixed" using fix_fmt() and returns its requirements for operands
2) The stream stack is searched for fitting streams and new ones are created if none match the requirements
3) return value properties are computed from the fixed operands
4) the return stream with the properties is pushed to the stream stack
"""


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
        op_list = ", ".join([op.name for op in ops])
        return "{}({})".format(self.name, op_list)

    def get_requirements(self):
        if self.name == DELAY:
            # require positive to throw no runtime_error, second stream arbitrary
            return [(POSITIVE, MAX_INTEGER, DONT_CARE, DONT_CARE, DONT_CARE), DONT_CARE_REQ]
        elif self.name == TIME:
            # input can be arbitrary, properties are propagated
            return [DONT_CARE_REQ]
        elif self.name == MERGE and self.types[0] == UNIT_STREAM:
            # types asserted by type check, rest arbitrary
            return [DONT_CARE_REQ, DONT_CARE_REQ]
        elif self.name == MERGE and self.types[0] == INT_STREAM:
            # both inputs must be int (=> type check), rest arbitrary
            return [DONT_CARE_REQ, DONT_CARE_REQ]
        elif self.name == LAST:
            # no properties over type check required
            return [DONT_CARE_REQ, DONT_CARE_REQ]
        elif self.name == COUNT:
            # no properties over type check required
            return [DONT_CARE_REQ]

    def get_return(self, ops):
        # returns (IS_POSITIVE, MAX_VAL, ZERO_TS, MAX_SIZE, MAX_TIMESTAMP)
        pos_flag_l, max_val_l, zero_ts_l, max_size_l, max_ts_l = ops[0].props
        if len(ops) == 2:
            pos_flag_r, max_val_r, zero_ts_r, max_size_r, max_ts_r = ops[1].props
            if self.name == DELAY:
                # streams has at most equal size and length as value stream
                return UNIT, DONT_CARE, False, max_size_l, max_ts_l
            elif self.name == MERGE and self.types[0] == UNIT_STREAM:
                # unit stream, zts_l || zts_r, sum of max sizes (at most), maximum of last timestamps
                return UNIT, DONT_CARE, zero_ts_r or zero_ts_l, max_size_l + max_size_r, max(max_ts_r, max_size_l)
            elif self.name == MERGE and self.types[0] == INT_STREAM:
                # int stream, zts_l || zts_r, sum of max sizes (at most), maximum of last timestamps
                if pos_flag_l == POSITIVE and pos_flag_r == POSITIVE:
                    # both inputs positive -> output positive
                    return POSITIVE, max(max_val_l, max_val_r), zero_ts_r or zero_ts_l, \
                           max_size_l + max_size_r, max(max_ts_r, max_size_l)
                else:
                    # otherwise non_positive
                    return NON_POSITIVE, DONT_CARE, zero_ts_r or zero_ts_l, \
                           max_size_l + max_size_r, max(max_ts_r, max_size_l)
            elif self.name == LAST:
                # follow trigger stream for all properties but value
                if pos_flag_l == POSITIVE:
                    return POSITIVE, max_val_l, zero_ts_r, max_size_r, max_ts_r
                else:
                    return NON_POSITIVE, DONT_CARE, zero_ts_r, max_size_r, max_ts_r
        else:
            # only one (l) input stream
            if self.name == COUNT:
                # max_size is new max_val (!), possibly non-positive if input has no event at ts 0
                if zero_ts_l:
                    return POSITIVE, max_size_l, True, max_size_l, max_ts_l
                else:
                    return NON_POSITIVE, DONT_CARE, True, max_size_l, max_ts_l
            elif self.name == TIME:
                # positive if no (!) event at timestamp 0, max_ts becomes new max_val
                if zero_ts_l:
                    return NON_POSITIVE, DONT_CARE, True, max_size_l, max_ts_l
                else:
                    return POSITIVE, max_ts_l, False, max_size_l, max_ts_l

    def __repr__(self):
        return self.name + "()"


class UnArithOp:
    def __init__(self):
        self.types = [INT_STREAM]
        self.ret_type = INT_STREAM
        self.op_int = None
        self.operator = None
        self.op_type = False

    # returns requirements
    def get_requirements(self):
        self.op_int = random.randint(INT_MIN, INT_MAX)
        self.operator = random.choice(ARITH_OPS)
        self.op_type = random.random() <= 0.5
        # prohibit overflows by constraining values
        if self.operator == "+":
            return [(POSITIVE, MAX_INTEGER - self.op_int, DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "*":
            return [(POSITIVE, MAX_INTEGER / self.op_int, DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "/":
            if self.op_type:
                return [DONT_CARE_REQ]
            else:
                return [(POSITIVE, DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "-":
            return [DONT_CARE_REQ]
        elif self.operator == "%":
            if self.op_type:
                return [DONT_CARE_REQ]
            else:
                return [(POSITIVE, DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE)]

    def get_return(self, ops):
        # returns (IS_POSITIVE, MAX_VAL, ZERO_TS, MAX_SIZE, MAX_TIMESTAMP)
        # zero_ts, max_size and max_ts are inherited from input, values are computed
        pos_flag, max_val, zero_ts, max_size, max_ts = ops[0].props
        if self.operator == "+":
            return POSITIVE, self.op_int + max_val, zero_ts, max_size, max_ts
        elif self.operator == "*":
            return POSITIVE, self.op_int * max_val, zero_ts, max_size, max_ts
        elif self.operator == "/":
            if self.op_type:
                # stream / op_int
                return NON_POSITIVE, DONT_CARE, zero_ts, max_size, max_ts
            else:
                return NON_POSITIVE, DONT_CARE, zero_ts, max_size, max_ts
        elif self.operator == "-":
            return NON_POSITIVE, DONT_CARE, zero_ts, max_size, max_ts
        elif self.operator == "%":
            return NON_POSITIVE, DONT_CARE, zero_ts, max_size, max_ts

    def get_stat(self, ops):
        assert (len(ops) == len(self.types))
        if self.op_type:
            return ARITH_FORMAT.format(ops[0].name, self.operator, self.op_int)
        else:
            return ARITH_FORMAT.format(self.op_int, self.operator, ops[0].name)
        pass

    def __repr__(self):
        return "unArithOp()"


class BinArithOp:
    def __init__(self):
        self.types = [INT_STREAM, INT_STREAM]
        self.ret_type = INT_STREAM
        self.operator = None

    def get_stat(self, ops):
        assert (len(ops) == len(self.types))
        self.operator = random.choice(ARITH_OPS)
        return ARITH_FORMAT.format(ops[0].name, self.operator, ops[1].name)

    def get_requirements(self):
        self.operator = random.choice(ARITH_OPS)
        if self.operator == "+":
            # require positive streams to compute positive one
            return [(POSITIVE, MAX_INTEGER / 2, DONT_CARE, DONT_CARE, DONT_CARE),
                    (POSITIVE, MAX_INTEGER / 2, DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "*":
            # require positive streams to compute positive one
            return [(POSITIVE, math.sqrt(MAX_INTEGER), DONT_CARE, DONT_CARE, DONT_CARE),
                    (POSITIVE, math.sqrt(MAX_INTEGER), DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "/":
            # first operand can be arbitrary since result must be non-positive anyway
            return [DONT_CARE_REQ, (POSITIVE, DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE)]
        elif self.operator == "-":
            # arbitrary operands since non-positive result in general
            return [DONT_CARE_REQ, DONT_CARE_REQ]
        elif self.operator == "%":
            # arbitrary first operand (non-positive result), second positive for definition purposes
            return [DONT_CARE_REQ,
                    (POSITIVE, DONT_CARE, DONT_CARE, DONT_CARE, DONT_CARE)]

    def get_return(self, ops):
        # returns (IS_POSITIVE, MAX_VAL, ZERO_TS, MAX_SIZE, MAX_TIMESTAMP)
        # zero_ts, max_size and max_ts are inherited from input, values are computed
        pos_flag_l, max_val_l, zero_ts_l, max_size_l, max_ts_l = ops[0].props
        pos_flag_r, max_val_r, zero_ts_r, max_size_r, max_ts_r = ops[1].props
        new_size = max_size_r + max_size_l
        new_zero_ts = zero_ts_l or zero_ts_r
        new_max_ts = max(max_ts_r, max_ts_l)
        if self.operator == "+":
            return POSITIVE, max_val_l + max_val_r, new_zero_ts, new_size, new_max_ts
        elif self.operator == "*":
            return POSITIVE, max_val_l * max_val_r, new_zero_ts, new_size, new_max_ts
        elif self.operator == "/":
            return NON_POSITIVE, DONT_CARE, new_zero_ts, new_size, new_max_ts
        elif self.operator == "-":
            return NON_POSITIVE, DONT_CARE, new_zero_ts, new_size, new_max_ts
        elif self.operator == "%":
            return NON_POSITIVE, DONT_CARE, new_zero_ts, new_size, new_max_ts

    def __repr__(self):
        return "binArithOp()"


class DefaultOp:
    def __init__(self):
        self.types = []
        self.ret_type = INT_STREAM
        self.op_int = 0

    def get_requirements(self):
        self.op_int = random.randint(INT_MIN, INT_MAX)
        return []

    def get_return(self, _ops):
        return POSITIVE, self.op_int, True, 1, 0

    def get_stat(self, _ops):
        return str(self.op_int)


class UnitOp:
    def __init__(self):
        self.types = []
        self.ret_type = UNIT_STREAM

    def get_requirements(self):
        return []

    def get_return(self, _ops):
        return UNIT, DONT_CARE, True, 1, 0

    def get_stat(self, _ops):
        return "()"


class Stream:
    def __init__(self, stream_type, is_input, properties):
        self.name = generate_name()
        self.is_input = is_input
        self.stream_type = stream_type
        self.props = properties

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
     Operation("delay", [INT_STREAM, STREAM], UNIT_STREAM),
    Operation("last", [INT_STREAM, STREAM], INT_STREAM),
    Operation("time", [STREAM], INT_STREAM),
    Operation("merge", [INT_STREAM, INT_STREAM], INT_STREAM),
    Operation("merge", [UNIT_STREAM, UNIT_STREAM], UNIT_STREAM),
    Operation("count", [STREAM], INT_STREAM),
    UnArithOp(),
    BinArithOp(),
     DefaultOp(),
     UnitOp()
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


def req_to_prop(requirements, stream_type, trace_length):
    # ony zero time stamp matters, rest is defined by input generation params
    _, _, r_zts, _, _ = requirements
    if r_zts == DONT_CARE:
        r_zts = False
    if stream_type == STREAM:
        # fix random stream type if only stream is required
        stream_type = UNIT_STREAM if random.random() < 0.5 else INT_STREAM
    if stream_type == UNIT_STREAM:
        return UNIT, DONT_CARE, r_zts, trace_length, trace_length
    elif stream_type == INT_STREAM:
        return POSITIVE, INT_MAX, r_zts, trace_length, trace_length
    else:
        raise RuntimeError("Undefined stream type {}".format(stream_type))
    pass


def req_match(stream_props, func_reqs):
    r_pos, r_max_val, r_zts, r_max_size, r_max_ts = func_reqs
    s_pos, s_max_val, s_zts, s_max_size, s_max_ts = stream_props
    ret_val = True
    # check positive flag and max val
    if r_pos == POSITIVE:
        if s_pos == POSITIVE:
            if r_max_val != DONT_CARE:
                ret_val = ret_val and s_max_val <= r_max_val
        else:
            ret_val = False

    # check zero time stamp
    if r_zts != DONT_CARE:
        ret_val = ret_val and r_zts == s_zts

    # check max_size (currently unused)
    if r_max_size != DONT_CARE:
        ret_val = ret_val and s_max_size <= r_max_size

        # check max timestamp (currently unused)
        if not r_max_size != DONT_CARE:
            ret_val = ret_val and s_max_ts <= r_max_ts

    return ret_val


"""
idea: make list of functions (?) and pick length elements. Then, lazily iterate through the list and get
as many streams as required, prefer to use last ones for propagation purposes (maybe non-deterministically /
decreased chance for elements lower in the "stack"?
"""


# get streams with fitting type in stream stack and possibly allocates new ones
def get_streams(stream_stack, stream_types, requirements, trace_length):
    not_found = copy.deepcopy(stream_types)
    not_found.reverse()
    requirements.reverse()
    op_streams = []
    for s in stream_stack[::-1]:
        if len(not_found) == 0:
            break
        if s.stream_type == not_found[-1] or not_found[-1] == STREAM:
            # check requirements
            if req_match(s.props, requirements[-1]):
                if random.random() < PICK_CHANCE:
                    op_streams.append(s)
                    not_found.pop()
                    requirements.pop()
    for i, t in enumerate(not_found):
        # if operands have not been found, alloc new input (!) streams
        props = req_to_prop(requirements[i], t, trace_length)
        op_streams.append(alloc_new(stream_stack, t, True, props))
    return op_streams


# alloc new stream in stream stack and return newly created stream
def alloc_new(stream_stack, stream_type, is_input, properties):
    pos, max_val, zts, max_size, max_ts = properties

    # assert that there is no unspecified behavior
    assert pos != DONT_CARE
    assert zts != DONT_CARE
    assert max_size != DONT_CARE
    assert max_ts != DONT_CARE
    assert not (pos == POSITIVE and max_val == DONT_CARE)

    if stream_type == STREAM:
        # infer actual stream type from properties
        stream_type = UNIT_STREAM if properties[0] == UNIT else INT_STREAM
    new_stream = Stream(stream_type, is_input, properties)
    stream_stack.append(new_stream)
    return new_stream


def generate_spec(length, seed, mode, trace_length):
    random.seed(seed)

    def_list = []
    stream_stack = []

    # generate list of length elements
    for _ in range(length):
        op = random.choice(OPERATIONS)
        requirements = op.get_requirements()
        operands = get_streams(stream_stack, op.types, requirements, trace_length)
        return_props = op.get_return(operands)
        new_stream = alloc_new(stream_stack, op.ret_type, False, return_props)
        def_list.append(DEF_FORMAT.format(new_stream.name, op.get_stat(operands)))

    # generate input stream statements
    in_list = []
    out_list = []
    int_names = []
    unit_names = []
    int_props = []
    unit_props = []
    for i, s in enumerate(stream_stack):
        if s.is_input:
            in_list.append(IN_FORMAT.format(s.name, s.type_str()))
            if s.stream_type == INT_STREAM:
                int_names.append(s.name)
                int_props.append(s.props)
            elif s.stream_type == UNIT_STREAM:
                unit_names.append(s.name)
                unit_props.append(s.props)
            else:
                raise RuntimeError("Found stream with invalid type {}".format(s.stream_type))
        else:
            if mode == MODE_DEBUG or i >= len(stream_stack) - 4:
                out_list.append(OUT_FORMAT.format(s.name, s.type_str()))

    stat_list = []
    stat_list.extend(in_list)
    stat_list.extend(def_list)
    stat_list.extend(out_list)
    return int_names, unit_names, int_props, unit_props, "\n".join(stat_list)


def random_string(str_size):
    return "".join(random.choice(string.ascii_letters) for _ in range(str_size))


def generate_test(spec_length, trace_length, seed, target_folder, mode):
    # generate specification
    int_names, unit_names, int_props, unit_props, spec = generate_spec(spec_length, seed, mode, trace_length)

    # generate int_weights and unit weights
    random.seed(seed)
    int_weights = []
    unit_weights = []
    for _ in int_names:
        int_weights.append(random.random())
    for _ in unit_names:
        unit_weights.append(random.random())

    # generate input trace
    trace = generate_property_trace(int_names, unit_names, int_props, unit_props,
                                    int_weights, unit_weights, INT_MIN, INT_MAX, trace_length, seed)

    # write to files
    test_name = "test_{}_sl{}_tl{}_{}".format(seed, spec_length, trace_length, mode)
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
    parser.add_argument("-m", "--mode", help="Mode to be run in (debug := all intermediate output streams, "
                                             "benchmark := max 3 output streams)", choices=[MODE_DEBUG, MODE_BENCHMARK],
                        default=MODE_DEBUG)

    args = parser.parse_args()

    # set seed
    seed = random_string(SEED_SIZE) if args.seed is None else args.seed

    out_folder = args.output_folder
    mode = args.mode

    # create output folder if it does not exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    for _ in range(args.num_tests):
        generate_test(args.spec_length, args.trace_length, seed, out_folder, mode)
        # get next seed
        random.seed(seed)
        seed = random_string(SEED_SIZE)


if __name__ == "__main__":
    main()
