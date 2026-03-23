import os
import re
import subprocess
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_NODES_DIR = os.path.join(ROOT_DIR, "custom_nodes")
REQUIREMENTS_FILE = os.path.join(ROOT_DIR, "requirements.txt")
CUSTOM_NODE_FILE = os.path.join(CUSTOM_NODES_DIR, "custom_node.txt")


def run(cmd, cwd=None):
    print(f"  > {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def install_requirements():
    print("\n=== Installing main requirements ===")
    run([sys.executable, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])
    print("Done.\n")


def parse_custom_nodes(filepath):
    """
    Parse custom_node.txt and return list of dicts:
        {"clone_url": str, "req_file": str | None}
    """
    entries = []
    current_url = None

    with open(filepath, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Skip empty lines and section comments (# N. ...)
            if not line or re.match(r"^#", line):
                current_url = None
                continue

            if line.startswith("git clone "):
                current_url = line.split()[-1]  # last token is the URL

            elif line.startswith("pip install -r ") and current_url:
                req_path = line.split()[-1]  # e.g. ComfyUI-Manager/requirements.txt
                entries.append({"clone_url": current_url, "req_file": req_path})
                current_url = None

    return entries


def install_custom_nodes():
    print("=== Installing custom nodes ===\n")
    os.makedirs(CUSTOM_NODES_DIR, exist_ok=True)

    entries = parse_custom_nodes(CUSTOM_NODE_FILE)

    for entry in entries:
        url = entry["clone_url"]
        # Derive repo name from URL (strip trailing .git if present)
        repo_name = url.rstrip("/").split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        repo_path = os.path.join(CUSTOM_NODES_DIR, repo_name)

        print(f"--- {repo_name} ---")

        # Clone only if not already present
        if os.path.isdir(repo_path):
            print(f"  Already exists, skipping clone.\n")
        else:
            print(f"  Cloning {url} ...")
            try:
                run(["git", "clone", url, repo_path])
            except subprocess.CalledProcessError as e:
                print(f"  ERROR: git clone failed for {url}: {e}\n")
                continue

        # Install node's requirements if the file exists
        if entry["req_file"]:
            # req_file path is relative to custom_nodes dir (e.g. "ComfyUI-Manager/requirements.txt")
            req_abs = os.path.join(CUSTOM_NODES_DIR, entry["req_file"])
            if os.path.isfile(req_abs):
                print(f"  Installing {entry['req_file']} ...")
                try:
                    run([sys.executable, "-m", "pip", "install", "-r", req_abs])
                except subprocess.CalledProcessError as e:
                    print(f"  WARNING: pip install failed: {e}")
            else:
                print(f"  requirements.txt not found at {req_abs}, skipping.")

        print()

    print("=== Custom nodes setup complete ===\n")


if __name__ == "__main__":
    install_requirements()
    install_custom_nodes()
    print("All done!")
