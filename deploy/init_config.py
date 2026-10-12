"""Initialize private deployment files without overwriting existing settings."""
from pathlib import Path
import secrets


def initialize(root: Path) -> bool:
    config = root / "config"
    config.mkdir(mode=0o700, exist_ok=True)
    for name in ("data", "artifacts"):
        (root / name).mkdir(exist_ok=True)
    target = config / ".env"
    if target.exists():
        print("config/.env already exists; left unchanged.")
        return False
    template = (root / ".env.example").read_text(encoding="utf-8")
    for name in ("CITETRACE_ADMIN_TOKEN", "CITETRACE_USER_TOKEN"):
        template = template.replace(name + "=\n", name + "=" + secrets.token_urlsafe(32) + "\n")
    with target.open("x", encoding="utf-8") as handle:
        target.chmod(0o600)
        handle.write(template)
    print("Created config/.env with random login tokens. Edit it privately to add API keys.")
    return True


if __name__ == "__main__":
    initialize(Path(__file__).resolve().parents[1])
