#!/usr/bin/python3 -u

import shutil
import argparse
import os
import subprocess
import re
import colorama

TESSLA = "tessla.jar"
TESSLA_DL = "https://git.tessla.io/tessla/tessla/builds/artifacts/master/raw/target/scala-2.13/tessla-assembly-1.2.2" \
            ".jar?job=deploy "
WINDER = "winder"
ARC = "arc"
BIN = "bin"
DATA = "data/"
OUT_EXT = ".out"

ARC_OUT = os.path.join(DATA, "arc.out")
TESSLA_OUT = os.path.join(DATA, "tessla.out")

WINDER_OUT_PATTERN = r"Compiled successfully to (.*\.coil)"


def magenta(text):
    return colorama.Fore.MAGENTA + text + colorama.Fore.RESET


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


def run_tessla(spec, infile):
    os.system("java -jar {} interpreter {} {} > {} ".format(os.path.join(BIN, TESSLA), os.path.join(DATA, spec),
                                                            os.path.join(DATA, infile), TESSLA_OUT))


def run_arc(spec, infile):
    # compile spec
    p = subprocess.Popen([os.path.join(BIN, WINDER), "compile", os.path.join(DATA, spec)], stdout=subprocess.PIPE)

    output = p.stdout.read().decode()
    matches = re.match(WINDER_OUT_PATTERN, output)
    if matches is None:
        print("Problem during .coil compilation, exiting.")
        exit(1)

    coil_file = matches.group(1)

    command = "{} -s {} {}".format(os.path.join(BIN, ARC), coil_file, os.path.join(DATA, infile))
    os.system(command)

    shutil.move(os.path.join(DATA, infile[:infile.find(".")]) + OUT_EXT, ARC_OUT)

    # remove .coil file
    os.remove(coil_file)


def check_diff():
    exit_code = os.system("diff {} {}".format(TESSLA_OUT, ARC_OUT))
    if exit_code == 0:
        print_green("Comparison successful, no differences found")
        return 0
    else:
        print_red("Differences in output were found, see above output.")
        return exit_code


def compare(spec, infile):
    print_delimiter()
    print_bright("Running comparison for specification {} and input file {}...\n".format(magenta(spec),
                                                                                         magenta(infile)))
    run_arc(spec, infile)
    run_tessla(spec, infile)
    exit_code = check_diff()
    if exit_code == 0:
        os.remove(ARC_OUT)
        os.remove(TESSLA_OUT)
    exit(exit_code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", help="Whether to rebuild binaries", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(BIN):
        print("{} folder not found, rebuilding binaries".format(BIN))
        os.mkdir(BIN)
        rebuild()
    if args.rebuild:
        rebuild()

    spec = "basictest.tessla"
    infile = "basictest.in"
    compare(spec, infile)


if __name__ == "__main__":
    main()
