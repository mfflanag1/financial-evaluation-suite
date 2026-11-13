"""Private component entrypoints share installed, source-checked supervision."""
import sys


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "component":
        from .hooks import component_main
        component_main(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "child":
        from .launcher import child_launch
        child_launch(sys.argv[2])
    else:
        from .launcher import main as cli
        cli()


if __name__ == "__main__":
    main()
