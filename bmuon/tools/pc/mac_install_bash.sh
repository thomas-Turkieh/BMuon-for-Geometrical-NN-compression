#!/bin/bash
# macOS without Homebrew: builds GNU bash 5 into ~/.local/bin (the night guard needs bash >= 4,
# macOS ships bash 3.2). Needs the Xcode Command Line Tools (xcode-select --install). About 2 min.
#   bash bmuon/tools/pc/mac_install_bash.sh [VERSION]      (default 5.2.37, from ftp.gnu.org)
set -eu
V="${1:-5.2.37}"
PREFIX="$HOME/.local"
if [ -x "$PREFIX/bin/bash" ] && "$PREFIX/bin/bash" -c '[ "${BASH_VERSINFO[0]}" -ge 4 ]'; then
  echo "already installed: $("$PREFIX/bin/bash" --version | head -1)"; exit 0
fi
xcode-select -p > /dev/null 2>&1 || { echo "Command Line Tools missing: run xcode-select --install"; exit 1; }
tmp=$(mktemp -d)
trap 'echo "build failed, log in $tmp/bash-$V/build.log"' ERR
curl -fsSL "https://ftp.gnu.org/gnu/bash/bash-$V.tar.gz" | tar -xz -C "$tmp"
cd "$tmp/bash-$V"
./configure --prefix="$PREFIX" --without-bash-malloc > build.log 2>&1
make -j"$(sysctl -n hw.ncpu)" >> build.log 2>&1
make install >> build.log 2>&1
trap - ERR
cd / && rm -rf "$tmp"
"$PREFIX/bin/bash" --version | head -1
