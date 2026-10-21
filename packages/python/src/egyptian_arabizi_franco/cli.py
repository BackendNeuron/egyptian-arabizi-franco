"""egyptian-arabizi-franco "ezayak 3amel eh"     convert the arguments
   echo "yalla bina" | egyptian-arabizi-franco   or each line from stdin"""

import sys


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic on Windows consoles
    from . import convert

    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print(__doc__)
            return
        print(convert(" ".join(sys.argv[1:])))
        return
    lines = [line.rstrip("\n") for line in sys.stdin]
    for out in convert(lines):
        print(out)


if __name__ == "__main__":
    main()
