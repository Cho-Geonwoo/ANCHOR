#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external

checkout_dependency() {
  local name="$1" url="$2" revision="$3"
  if [[ ! -d "external/$name/.git" ]]; then
    git clone "$url" "external/$name"
  fi
  local current
  current="$(git -C "external/$name" rev-parse HEAD)"
  if [[ "$current" != "$revision" ]]; then
    if [[ -n "$(git -C "external/$name" status --porcelain)" ]]; then
      echo "Local changes in external/$name; use a fresh directory." >&2
      exit 1
    fi
    git -C "external/$name" checkout --detach "$revision"
  fi
  git -C "external/$name" submodule update --init --recursive
  python -m pip install --no-deps --no-build-isolation -e "external/$name"
}

checkout_dependency mjrl https://github.com/aravindr93/mjrl.git 3871d93763d3b49c4741e6daeaebbc605fe140dc
checkout_dependency D4RL https://github.com/zhouzypaul/D4RL.git d005dca9dab0c0c50aa5aa155cdd6b64de1cfe4d
checkout_dependency mj_envs https://github.com/nakamotoo/mj_envs.git 88fadb11c38e8141dfe7bbde92a8954858c4b9f2
