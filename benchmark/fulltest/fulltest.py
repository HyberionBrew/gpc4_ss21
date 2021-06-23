#!/usr/bin/python3 -u

import shutil
import argparse
import os
import subprocess
import re
import colorama
import time

TESSLA = "tessla.jar"
TESSLA_DL = "https://git.tessla.io/tessla/tessla/builds/artifacts/master/raw/target/scala-2.13/tessla-assembly-1.2.2" \
            ".jar?job=deploy "
WINDER = "winder"
ARC = "arc"
SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))
BIN = os.path.join(SCRIPT_DIR, "bin")
DATA = os.path.join(SCRIPT_DIR, "data")
OUT_EXT = ".out"

ARC_OUT = os.path.join(DATA, "arc.out")
TESSLA_OUT = os.path.join(DATA, "tessla.out")

WINDER_OUT_PATTERN = r"Compiled successfully to (.*\.coil)"

MODE_SEQUENTIAL = 252
MODE_CUDA = 253
MODE_CUDA_SM = 254
MODE_THRUST = 255

MODE_MAP = {
    "seq": MODE_SEQUENTIAL,
    "cuda": MODE_CUDA,
    "cuda-sm": MODE_CUDA_SM,
    "thrust": MODE_THRUST
}


def get_arc_mode(mode):
    if mode == MODE_SEQUENTIAL:
        return "-s"
    elif mode == MODE_CUDA:
        return ""
    else:
        print_red("Unsupported mode \"{}\", exiting".format(name_str(mode)))


def name_str(global_var):
    names = [name for name in globals() if globals()[name] is global_var]
    return names[0]


def magenta(text):
    return colorama.Fore.MAGENTA + text + colorama.Fore.RESET


def red(text):
    return colorama.Fore.RED + text + colorama.Fore.RESET


def print_bright(text):
    print(colorama.Style.BRIGHT + text + colorama.Style.RESET_ALL)


def print_red(text):
    print_bright(colorama.Fore.RED + text)


def print_green(text):
    print_bright(colorama.Fore.GREEN + text)


def print_delimiter():
    print_bright("\n================================================================\n")


def rebuild():
    print_bright("Rebuilding...\n")

    print("Rebuilding winder...")
    # build winder
    winder_ec = os.system("cd ../../winder; cargo build --release")
    # copy binary to ./bin
    shutil.copy("../../winder/target/release/winder", "{}/{}".format(BIN, WINDER))
    print()

    print("Rebuilding arc...")
    # build arc
    arc_ec = os.system("cd ../../arc; cmake . > /dev/null 2>&1; cmake --build . --target arc ")
    # copy binary to ./bin
    shutil.copy("../../arc/arc", "{}/{}".format(BIN, ARC))
    print()

    tessla_ec = 0
    # get TeSSLa from web (for now)
    if not os.path.exists(os.path.join(BIN, TESSLA)):
        print("Re-downloading TeSSLa...")
        tessla_ec = os.system("cd {}; wget -O {} {}".format(BIN, TESSLA, TESSLA_DL))

    if winder_ec != 0 or tessla_ec != 0 or arc_ec != 0:
        print_red("Rebuilding failed, see above output.")
        exit(1)
    else:
        print_green("Successfully finished rebuilding")
    print_delimiter()


def run_tessla(spec, infile):
    start = time.time()
    print("{}: Running interpreter".format(TESSLA))
    tessla_ec = os.system("java -jar {} interpreter {} {} > {} ".format(os.path.join(BIN, TESSLA),
                                                                        spec,
                                                                        infile, TESSLA_OUT))
    tessla_et = round((time.time() - start)*1000)
    if tessla_ec != 0:
        print_red("{}: Problem during execution, exiting".format(TESSLA))
    else:
        print("{}: Execution successful ({} ms)\n".format(TESSLA, tessla_et))


def run_arc(spec, infile, mode):
    print("{}: Compiling .coil file...".format(WINDER))

    # compile spec
    start = time.time()
    p = subprocess.Popen([os.path.join(BIN, WINDER), "compile", spec], stdout=subprocess.PIPE)

    p.wait()
    winder_et = round((time.time()-start)*1000)
    output = p.stdout.read().decode()
    winder_ec = p.returncode
    matches = re.match(WINDER_OUT_PATTERN, output)
    if matches is None or winder_ec != 0:
        print_red("{}: Problem during .coil compilation, exiting.".format(WINDER))
        exit(1)
    else:
        print("{}: Compilation successful ({} ms)\n".format(WINDER, winder_et))

    coil_file = matches.group(1)

    arc_mode = get_arc_mode(mode)
    command = "{} -o {} {} {} {}".format(os.path.join(BIN, ARC), ARC_OUT, arc_mode, coil_file, infile)
    print("{}: Running interpreter in mode {}".format(ARC, magenta(name_str(mode))))
    start = time.time()
    arc_ec = os.system(command)
    arc_et = round(((time.time()-start)*1000))
    if arc_ec != 0:
        print_red("{}: Problem during execution, exiting.".format(ARC))
        exit(1)
    else:
        print("{}: Execution successful ({} ms, total: {} ms) \n".format(ARC, arc_et, winder_et + arc_et))

    # remove .coil file
    os.remove(coil_file)


def check_diff():
    print("Comparing output files...")
    exit_code = os.system("diff <(sort {}) <(sort {})".format(TESSLA_OUT, ARC_OUT))
    if exit_code == 0:
        print_green("Comparison successful, no differences found")
        return 0
    else:
        print_red("Differences in output were found, see above output (output files \"{}\" and \"{}\")"
                  .format(TESSLA_OUT, ARC_OUT))
        return exit_code


def compare(spec, infile, arc_mode):
    print_bright("Running comparison for specification {} and input file {}...\n".format(magenta(spec),
                                                                                         magenta(infile)))
    run_arc(spec, infile, arc_mode)
    run_tessla(spec, infile)
    exit_code = check_diff()
    if exit_code == 0:
        os.remove(ARC_OUT)
        os.remove(TESSLA_OUT)
    else:
        exit(exit_code)


def compare_all(arc_mode):
    # collect all files in data folder
    files = os.listdir(DATA)
    input_files = list(filter(lambda a: a.endswith(".in"), files))
    spec_files = list(filter(lambda a: a.endswith(".tessla"), files))
    all_list = []
    for inf in input_files:
        pref = inf.split(".")[0]
        spec = [s for s in spec_files if s.split(".")[0] == pref]
        if len(spec) > 0:
            all_list.append((os.path.join(DATA, spec[0]), os.path.join(DATA, inf)))
    c = 0
    for s, i in all_list:
        compare(s, i, arc_mode)
        c += 1
        if c != len(all_list):
            print_delimiter()


def main():
    parser = argparse.ArgumentParser(description="Run TeSSLa and winder+arc and compare outputs")
    parser.add_argument("--rebuild", help="Whether to rebuild binaries", action="store_true")
    parser.add_argument("-s", "--spec", help="Specification to run the comparison on")
    parser.add_argument("-i", "--input", help="Input file to run the comparison on")
    parser.add_argument("-m", "--mode", help="Running mode for arc vm", choices=["seq", "cuda", "cuda-sm", "thrust"],
                        default="seq")
    args = parser.parse_args()

    if args.spec is not None and args.input is None or args.input is not None and args.spec is None:
        print_red("Please specify either both input and specification or neither to run all comparisons")
        exit(1)

    mode = MODE_MAP[args.mode]

    if not os.path.exists(BIN):
        print(magenta("{} folder not found, rebuilding binaries".format(BIN)))
        os.mkdir(BIN)
        rebuild()
    elif args.rebuild:
        rebuild()

    if args.spec is not None:
        # run single comparison
        compare(args.spec, args.input, mode)
    else:
        # run all comparisons
        compare_all(mode)


if __name__ == "__main__":
    main()