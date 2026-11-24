"""Save an installed Windows environment as a portable, path-free version lock."""

import importlib.metadata
import json
import platform

from scripts.prepare_assets import ROOT


def main():
    if platform.system() != "Windows":
        raise RuntimeError("This lock documents the tested Windows environment")
    versions = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower() != "agelens-mlops"
    }
    lines = [
        "# Generated for Windows amd64, Python 3.12; not a Linux lock.",
        "# Use scripts/setup.ps1: bootstrap CPU PyTorch before installing this lock.",
        "--index-url https://pypi.org/simple",
    ]
    for name, version in sorted(versions.items(), key=lambda entry: entry[0].lower()):
        if name.lower() == "clip":
            lines.append(
                "clip @ https://github.com/openai/CLIP/archive/"
                "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.zip"
            )
        else:
            lines.append(f"{name}=={version}")
    (ROOT / "requirements-windows.lock").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "python": platform.python_version(),
        "os": platform.system(),
        "packages": versions,
        "trained": False,
    }
    output = ROOT / "artifacts/private/environment.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Locked {len(versions)} packages without local paths.")


if __name__ == "__main__":
    main()
