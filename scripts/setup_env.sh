#!/bin/sh
# Build the pinned environment used for every result in the paper.
#
#   sh scripts/setup_env.sh [python]
#
# `python` defaults to python3.12. The paper's environment A is Python 3.12.3;
# environment B is Python 3.12.14 with the same four library versions. numpy
# 1.26.4 has no wheels for Python 3.13 or later, so a 3.12 interpreter is
# required to reproduce the reported numbers.
#
# A virtual environment is used rather than conda because conda refuses to
# create an environment until the Anaconda channel terms of service are
# accepted, which is not a decision this script should make for you.
set -eu

PY="${1:-python3.12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"

command -v "$PY" >/dev/null 2>&1 || {
  echo "setup_env.sh: $PY not found. Pass the path to a Python 3.12 interpreter:" >&2
  echo "  sh scripts/setup_env.sh /usr/bin/python3.12" >&2
  exit 2
}

"$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' || {
  echo "setup_env.sh: $PY is not Python 3.12; numpy 1.26.4 will not install." >&2
  exit 2
}

"$PY" -m venv "$VENV"

# POSIX venvs put the interpreter in bin/, Windows venvs in Scripts/
if [ -x "$VENV/bin/python" ]; then
  VPY="$VENV/bin/python"
  ACTIVATE="$VENV/bin/activate"
else
  VPY="$VENV/Scripts/python.exe"
  ACTIVATE="$VENV/Scripts/activate"
fi

"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r "$ROOT/requirements.txt"
"$VPY" - <<'EOF'
import numpy, scipy, sklearn, pandas, sys
print("python       %s" % sys.version.split()[0])
print("numpy        %s" % numpy.__version__)
print("scipy        %s" % scipy.__version__)
print("scikit-learn %s" % sklearn.__version__)
print("pandas       %s" % pandas.__version__)
EOF

mkdir -p "$ROOT/data/exp" "$ROOT/data/exp12"
echo
echo "environment ready: $VENV"
echo "activate it with:  . $ACTIVATE"
