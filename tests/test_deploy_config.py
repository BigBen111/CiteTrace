import importlib.util
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("init_config", ROOT / "deploy/init_config.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_initialize_private_config_without_overwrite(tmp_path):
    (tmp_path / ".env.example").write_text((ROOT / ".env.example").read_text())
    assert module.initialize(tmp_path)
    target = tmp_path / "config/.env"
    first = target.read_text()
    values = dict(line.split("=", 1) for line in first.splitlines() if "=" in line)
    assert len(values["CITETRACE_ADMIN_TOKEN"]) >= 32
    assert values["CITETRACE_ADMIN_TOKEN"] != values["CITETRACE_USER_TOKEN"]
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not module.initialize(tmp_path)
    assert target.read_text() == first
    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "artifacts").is_dir()
