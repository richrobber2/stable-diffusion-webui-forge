import os
import sys
import subprocess
from modules.paths_internal import models_path, script_path, data_path, extensions_dir, extensions_builtin_dir, cwd  # noqa: F401


sys.path.insert(0, script_path)
sd_path = os.path.dirname(__file__)

def run_git(dir, name, command, desc=None, errdesc=None, custom_env=None, live=True):
    """Run a git command with error handling"""
    if desc is not None:
        print(desc)

    git = os.environ.get('GIT', "git")
    cmd = f'"{git}" -C "{dir}" {command}'
    
    try:
        return subprocess.check_output(cmd, shell=True, env=custom_env, stderr=subprocess.STDOUT).decode('utf-8')
    except subprocess.CalledProcessError as e:
        if errdesc is not None:
            print(f"{errdesc}", file=sys.stderr)
        raise e

def git_clone(url, dir, name, commithash=None):
    """Clone a git repository with optional commit hash checkout"""
    if os.path.exists(dir):
        if commithash is None:
            return

        current_hash = run_git(dir, name, 'rev-parse HEAD', None, f"Couldn't determine {name}'s hash").strip()
        if current_hash == commithash:
            return

        run_git(dir, name, f'fetch origin "{commithash}"', f"Fetching updates for {name}...", f"Couldn't fetch {name}")
        run_git(dir, name, f'checkout {commithash}', f"Checking out commit {commithash} for {name}...", f"Couldn't checkout commit {commithash} for {name}")
        return

    try:
        os.makedirs(os.path.dirname(dir), exist_ok=True)
        run_git(os.path.dirname(dir), name, f'clone "{url}" "{os.path.basename(dir)}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}")

        if commithash is not None:
            run_git(dir, name, f'checkout {commithash}', f"Checking out commit {commithash} for {name}...", f"Couldn't checkout commit {commithash} for {name}")
    except Exception as e:
        print(f"Error cloning repository {name}: {str(e)}", file=sys.stderr)
        if os.path.exists(dir):
            import shutil
            shutil.rmtree(dir, ignore_errors=True)
        raise e

# Repository configuration
repositories = {
    'BLIP': {
        'url': 'https://github.com/salesforce/BLIP.git',
        'path': os.path.join(sd_path, '../repositories/BLIP'),
        'must_exist': 'models/blip.py',
        'commit': None
    },
    'huggingface_guess': {
        'url': 'https://github.com/lllyasviel/huggingface_guess.git',
        'path': os.path.join(sd_path, '../repositories/huggingface_guess'),
        'must_exist': 'huggingface_guess/detection.py',
        'commit': None
    },
    'stable-diffusion-webui-assets': {
        'url': 'https://github.com/richrobber2/stable-diffusion-webui-assets.git',
        'path': os.path.join(sd_path, '../repositories/stable-diffusion-webui-assets'),
        'must_exist': 'favicon.ico',
        'commit': None
    }
}

# Repository and path setup
paths = {}

def ensure_repository(name, repo_info):
    """Ensure a repository is cloned and up to date"""
    try:
        git_clone(
            url=repo_info['url'],
            dir=repo_info['path'],
            name=name,
            commithash=repo_info['commit']
        )
        return True
    except Exception as e:
        print(f"Warning: Failed to setup {name} repository: {str(e)}", file=sys.stderr)
        return False

# Process repositories and set up paths
for name, repo_info in repositories.items():
    must_exist_path = os.path.join(repo_info['path'], repo_info['must_exist'])
    
    if not os.path.exists(must_exist_path):
        print(f"Repository {name} not found, attempting to clone...", file=sys.stderr)
        if ensure_repository(name, repo_info):
            print(f"Successfully set up {name} repository")
        else:
            print(f"Warning: {name} not found at path {must_exist_path}", file=sys.stderr)
            continue
    
    repo_path = os.path.abspath(repo_info['path'])
    sys.path.append(repo_path)
    paths[name] = repo_path

# Additional paths
path_dirs = [
    (os.path.join(sd_path, '../packages_3rdparty'), 'gguf/quants.py', 'packages_3rdparty', []),
]

for d, must_exist, what, options in path_dirs:
    must_exist_path = os.path.abspath(os.path.join(script_path, d, must_exist))
    if not os.path.exists(must_exist_path):
        print(f"Warning: {what} not found at path {must_exist_path}", file=sys.stderr)
    else:
        d = os.path.abspath(d)
        if "atstart" in options:
            sys.path.insert(0, d)
        else:
            sys.path.append(d)
        paths[what] = d
