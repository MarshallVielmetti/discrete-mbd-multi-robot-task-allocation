import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


class ExternalSolverSetupTest(unittest.TestCase):
    def test_initialized_submodule_is_not_cloned_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = root / "scripts"
            scripts.mkdir()
            script = scripts / "setup_external_solvers.sh"
            shutil.copy2(Path(__file__).resolve().parents[1] / "scripts" / script.name, script)
            ctas = root / "external/resilient_team_planner"
            ctas.mkdir(parents=True)
            # Initialized submodules use a .git file rather than a directory.
            (ctas / ".git").write_text("gitdir: ../../.git/modules/ctas\n")
            (root / "external/yaml-cpp/.git").mkdir(parents=True)
            include = root / "gurobi/include"
            include.mkdir(parents=True)
            (include / "gurobi_c.h").touch()
            binaries = root / "bin"
            binaries.mkdir()
            for name, content in {
                "git": '#!/bin/sh\n[ "$1" != clone ] || exit 44\nexit 0\n',
                "cmake": "#!/bin/sh\nexit 0\n",
                "g++": "#!/bin/sh\nexit 0\n",
            }.items():
                path = binaries / name
                path.write_text(content)
                path.chmod(0o755)
            environment = dict(os.environ, GUROBI_HOME=str(include.parent),
                               PATH=str(binaries) + os.pathsep + os.environ["PATH"])
            for _ in range(2):
                process = subprocess.run([str(script)], env=environment, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
