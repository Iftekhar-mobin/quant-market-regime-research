"""Build and push the Hugging Face Space for this project.

The Space runs as a Docker Space: Hugging Face retired the managed Streamlit
SDK, so the console is containerised and serves on the port declared in
README.md.

The Space is a curated subset of the repository: the library, the console, the
configuration, the sample datasets, and one pre-computed study so the Results
and Signals tabs have something to show on a first visit. Full market history,
experiment archives and logs are left behind.

Prerequisites
-------------
Log in once, in your own terminal, so the token never passes through anything
else:

    huggingface-cli login

Then:

    python deploy/huggingface/deploy_space.py --repo <your-username>/quant-market-regime-research

Add --dry-run to build the payload and inspect it without uploading.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# Everything the Space needs, and nothing else.
INCLUDE_DIRS = ["src", "app", "configs", "data/samples", "docs"]
INCLUDE_FILES = ["LICENSE"]

# One stored study, so the Results / Model comparison / Signals tabs are
# populated the moment someone opens the Space.
SEED_EXPERIMENT = "book_trace"


def build(staging: Path) -> None:
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for rel in INCLUDE_DIRS:
        source = ROOT / rel
        if not source.exists():
            print(f"  ! missing {rel}, skipping")
            continue
        shutil.copytree(
            source,
            staging / rel,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints"),
        )
        print(f"  + {rel}")

    for rel in INCLUDE_FILES:
        if (ROOT / rel).exists():
            shutil.copy2(ROOT / rel, staging / rel)
            print(f"  + {rel}")

    # Streamlit theme, so the Space looks like the local console.
    if (ROOT / ".streamlit").exists():
        shutil.copytree(ROOT / ".streamlit", staging / ".streamlit")
        print("  + .streamlit")

    # Space metadata README (the YAML front matter is what configures the Space).
    shutil.copy2(HERE / "README.md", staging / "README.md")
    shutil.copy2(HERE / "requirements.txt", staging / "requirements.txt")
    shutil.copy2(HERE / "Dockerfile", staging / "Dockerfile")
    print("  + README.md, requirements.txt, Dockerfile")

    # Writable directories the pipeline expects to exist.
    for rel in ("data/raw", "data/processed", "experiments", "reports"):
        (staging / rel).mkdir(parents=True, exist_ok=True)
        (staging / rel / ".gitkeep").touch()

    seed = ROOT / "experiments" / SEED_EXPERIMENT
    if seed.exists():
        shutil.copytree(seed, staging / "experiments" / SEED_EXPERIMENT)
        print(f"  + experiments/{SEED_EXPERIMENT} (pre-computed study)")
    else:
        print(f"  ! no seed experiment at {seed}; Results tab will start empty")

    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"\nStaged {total / 1e6:.1f} MB in {staging}")


def push(staging: Path, repo_id: str, private: bool) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is not installed:  pip install -U huggingface_hub")
        return 1

    api = HfApi()
    try:
        who = api.whoami()
    except Exception:
        print(
            "Not logged in to Hugging Face.\n"
            "Run this in your own terminal first, so the token stays with you:\n\n"
            "    huggingface-cli login\n"
        )
        return 1

    print(f"Logged in as {who.get('name', '?')}")

    api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=private,
        exist_ok=True,
    )
    print(f"Space ready: https://huggingface.co/spaces/{repo_id}")

    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="space",
        commit_message="Deploy research console",
        ignore_patterns=["__pycache__", "*.pyc"],
    )
    print(f"\nUploaded. The Space will build in a few minutes:\n"
          f"  https://huggingface.co/spaces/{repo_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="<username>/<space-name>")
    parser.add_argument("--private", action="store_true", help="create a private Space")
    parser.add_argument("--dry-run", action="store_true", help="build only, do not upload")
    args = parser.parse_args()

    staging = HERE / "_staging"
    print("Building Space payload...")
    build(staging)

    if args.dry_run:
        print("\nDry run: nothing uploaded.")
        return 0
    if not args.repo:
        print("\nPass --repo <username>/<space-name> to upload, or --dry-run to stop here.")
        return 1
    return push(staging, args.repo, args.private)


if __name__ == "__main__":
    sys.exit(main())
