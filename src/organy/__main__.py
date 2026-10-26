# This script is the main entrypoint of Organy application.
import sys

from organy.cli import main

# argparse automatically defaults to `sys.argv` if argv is None.
sys.exit(main())
