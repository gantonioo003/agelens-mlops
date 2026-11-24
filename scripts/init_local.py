"""Create local directories and credentials without overwriting existing secrets."""

import secrets

from scripts.prepare_assets import ROOT


def main():
    for directory in [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/manifests",
        "models",
        "artifacts/private",
        "runtime",
    ]:
        (ROOT / directory).mkdir(parents=True, exist_ok=True)
    path = ROOT / ".env"
    try:
        with path.open("x", encoding="utf-8") as out:
            out.write(f"POSTGRES_PASSWORD={secrets.token_hex(32)}\n")
            out.write("MLFLOW_TRACKING_URI=http://127.0.0.1:15000\n")
        print("Created private local .env; existing credentials are never overwritten.")
    except FileExistsError:
        print("Kept existing .env.")


if __name__ == "__main__":
    main()
