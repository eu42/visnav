#!/usr/bin/env bash

set -e

find ./include ./src ./test/src -iname "*.hpp" -or -iname "*.h" -or -iname "*.cpp" | xargs clang-format -i
