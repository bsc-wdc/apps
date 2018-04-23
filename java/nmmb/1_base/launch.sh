#!/bin/bash -e

  # Define script directory for relative calls
  SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

  # Call NMMB enqueue handler
  "${SCRIPT_DIR}"/JOB/enqueue_NMMB_MN.sh "$@"

