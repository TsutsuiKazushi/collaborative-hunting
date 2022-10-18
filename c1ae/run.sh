#!/bin/bash

option=aaa

help_message=$(cat << EOF
Usage: $0 [Options] <required>

Options:
    --option: indiv or share  (default=${option})
    --required: 4.2, 3.6, 3.0, 2.4, or 1.8
EOF
)
. utils/parse_options.sh

# Variables that must be specified are defined after parse_options.sh
required=$1

if [ $# -ne 1 ]; then
    echo "${help_message}"
    exit 1;
fi

set -euo pipefail

echo "option = ${option}"
echo "required = ${required}"

python -u c1ae.py ${required}
