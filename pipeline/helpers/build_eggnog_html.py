#!/usr/bin/env python3
import sys
import pandas as pd
import numpy as np
import json
import struct
from pathlib import Path

# We just call the main logic of N6_build_html_explorer, but we modify it to accept eggnog paths!
# But actually, N6_build_html_explorer.py is quite large.
# Let's just modify build_html_explorer.py temporarily or call it via sub-process.
