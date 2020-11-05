#!/bin/bash

scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
"$scriptdir/install_orthoseg.sh" --envname orthosegdev --fordev Y
