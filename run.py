#!/usr/bin/env python3
"""Quick launcher script for Just Dance Video Processor."""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from just_dance.main import main

if __name__ == "__main__":
    main()
