#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
external_root="${repo_root}/external"
ctas_dir="${external_root}/resilient_team_planner"
yaml_dir="${external_root}/yaml-cpp"
yaml_install="${external_root}/install/yaml-cpp"
ctas_patch="${repo_root}/patches/resilient_team_planner-linux.patch"
ctas_commit="6bc762c69e0423be1db0b8c855b955b22373e8ff"
yaml_commit="0579ae3d976091d7d664aa9d2527e0d0cff25763"
gurobi_root="${GUROBI_HOME:-${external_root}/gurobi1003/linux64}"

for command_name in git cmake g++; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

if [[ ! -f "${gurobi_root}/include/gurobi_c.h" ]]; then
    echo "Gurobi headers not found under ${gurobi_root}." >&2
    echo "Set GUROBI_HOME to the Gurobi platform directory (for example, /opt/gurobi1003/linux64)." >&2
    exit 1
fi

mkdir -p "${external_root}" "${external_root}/install"

if [[ ! -e "${ctas_dir}/.git" ]]; then
    git clone https://gitlab.com/barton-research-group/open/resilient_team_planner.git "${ctas_dir}"
fi
if ! git -C "${ctas_dir}" cat-file -e "${ctas_commit}^{commit}" 2>/dev/null; then
    git -C "${ctas_dir}" fetch origin "${ctas_commit}"
fi
git -C "${ctas_dir}" checkout --detach "${ctas_commit}"

if git -C "${ctas_dir}" apply --reverse --check --ignore-space-change --ignore-whitespace "${ctas_patch}" >/dev/null 2>&1; then
    echo "CTAS-D Linux build patch is already applied."
elif git -C "${ctas_dir}" apply --check --ignore-space-change --ignore-whitespace "${ctas_patch}" >/dev/null 2>&1; then
    git -C "${ctas_dir}" apply --ignore-space-change --ignore-whitespace "${ctas_patch}"
else
    echo "CTAS-D checkout has incompatible local changes; refusing to overwrite them." >&2
    exit 1
fi

if [[ ! -d "${yaml_dir}/.git" ]]; then
    git clone https://github.com/jbeder/yaml-cpp.git "${yaml_dir}"
fi
if ! git -C "${yaml_dir}" cat-file -e "${yaml_commit}^{commit}" 2>/dev/null; then
    git -C "${yaml_dir}" fetch origin "${yaml_commit}"
fi
git -C "${yaml_dir}" checkout --detach "${yaml_commit}"

# Gurobi's Linux C++ archive uses the old libstdc++ string ABI, so yaml-cpp and
# the CTAS-D executable must be built with the same setting.
cmake -S "${yaml_dir}" -B "${yaml_dir}/build" \
    -DYAML_CPP_BUILD_TESTS=OFF \
    -DYAML_CPP_BUILD_TOOLS=OFF \
    -DYAML_BUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
    -DCMAKE_INSTALL_PREFIX="${yaml_install}"
cmake --build "${yaml_dir}/build" --parallel
cmake --install "${yaml_dir}/build"

cmake -S "${ctas_dir}" -B "${ctas_dir}/build" \
    -DTEAMPLANNER_PYTOOL=OFF \
    -DGUROBI_HOME="${gurobi_root}" \
    -DCMAKE_PREFIX_PATH="${yaml_install}" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "${ctas_dir}/build" --parallel

echo "Built CTAS-D: ${ctas_dir}/build/main"
echo "A valid Gurobi license is still required at runtime."
