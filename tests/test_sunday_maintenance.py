import os
import shutil
import subprocess
from pathlib import Path


def test_sunday_maintenance_stops_before_retrain_when_refresh_fails(tmp_path):
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (repo / "execution" / "logs").mkdir(parents=True)

    src = Path(__file__).resolve().parents[1] / "scripts" / "sunday_maintenance.sh"
    dst = scripts / "sunday_maintenance.sh"
    shutil.copy2(src, dst)
    dst.chmod(0o755)

    fake_python = tmp_path / "fake_python.sh"
    fake_python.write_text("#!/bin/bash\necho fake update failed\nexit 7\n")
    fake_python.chmod(0o755)

    env = {**os.environ, "PYTHON": str(fake_python)}
    result = subprocess.run(
        [str(dst)], cwd=str(repo), env=env, capture_output=True, text=True, timeout=20
    )

    assert result.returncode == 7
    assert "Phase 1: Refreshing parquet store from IBKR" in result.stdout
    assert "Phase 2: Retraining stale ML models" not in result.stdout
