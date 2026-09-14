#!/usr/bin/env bash
# ==============================================================================
# Script 0: Download COUNT software
#
# Purpose:
#   Downloads COUNT (CountXXV.jar), a standard phylogenetic analysis tool
#   (Csűrös 2010, Bioinformatics), which we use for ancestral parsimony reconstruction.
#
# Output:
#   tools/CountXXV.jar
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$ROOT_DIR/tools"
JAR_PATH="$TOOLS_DIR/CountXXV.jar"
JAR_URL="https://github.com/csurosm/count/releases/download/v25.12-beta/CountXXV.jar"

# 1. Create tools directory if it does not exist
mkdir -p "$TOOLS_DIR"

# 2. Download COUNT JAR release if not already present
if [[ -f "$JAR_PATH" ]]; then
  echo "COUNT is already installed at: $JAR_PATH"
else
  echo "Downloading COUNT from: $JAR_URL"
  curl -sL -o "$JAR_PATH" "$JAR_URL"
  echo "Saved to: $JAR_PATH"
fi

# 3. Verify that the JAR executes with Java
echo "Testing COUNT execution:"
java -cp "$JAR_PATH" count.model.Parsimony 2>&1 | head -5 || true
