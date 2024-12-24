"""
This module provides functionality to save and restore the state of a WebUI and its extensions
to specific git commits, ensuring you can revert to a previously known good configuration.
"""

import os
import json
from datetime import datetime
from tqdm import tqdm
import git

from modules import shared, extensions, errors
from modules.paths_internal import script_path, config_states_dir

# This dictionary maps a human-readable name (like "Config: 2024-12-12 10:00:00")
# to the actual configuration dictionary loaded from file.
all_config_states = {}


def load_config_file(path):
    """Helper to load configuration from JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f'[ERROR]: Failed to load config state from {path}, reason: {e}')


def list_config_states():
    """
    Lists all saved configuration states.

    Each configuration state is stored as a JSON file in `config_states_dir`.
    The returned dictionary `all_config_states` maps a human-readable string
    (e.g., "Config: 2024-12-12 10:00:00") to the corresponding configuration data
    loaded from the JSON file.

    Returns:
        dict: A dictionary with keys as human-readable state names and values as configuration dicts.
    """
    global all_config_states
    all_config_states.clear()

    # Ensure the directory exists
    os.makedirs(config_states_dir, exist_ok=True)

    config_states = []

    # Iterate over all JSON files in the config_states_dir
    for filename in os.listdir(config_states_dir):
        if filename.endswith(".json"):
            path = os.path.join(config_states_dir, filename)
            config_data = load_config_file(path)  # replaced load logic
            if not config_data:
                continue

            # Ensure that "created_at" exists in the config
            assert "created_at" in config_data, '"created_at" does not exist in the config state file'

            # Add the file path to the config data for reference
            config_data["filepath"] = path
            config_states.append(config_data)

    # Sort by creation time, newest first
    config_states.sort(key=lambda cs: cs["created_at"], reverse=True)

    # Create a human-readable name for each configuration and populate all_config_states
    for cs in config_states:
        timestamp_str = datetime.fromtimestamp(cs["created_at"]).strftime('%Y-%m-%d %H:%M:%S')
        state_name = cs.get("name", "Config")
        full_name = f"{state_name}: {timestamp_str}"
        all_config_states[full_name] = cs

    return all_config_states


def get_webui_config():
    """
    Gathers the current WebUI's git configuration details.

    Returns:
        dict: A dictionary containing:
              - remote: The URL of the remote repository (if any)
              - commit_hash: The current commit hash
              - commit_date: The timestamp of the current commit
              - branch: The currently active branch name
    """
    webui_repo = None
    webui_remote = None
    webui_commit_hash = None
    webui_commit_date = None
    webui_branch = None

    # Check if the WebUI directory is a git repository
    if os.path.exists(os.path.join(script_path, ".git")):
        try:
            webui_repo = git.Repo(script_path)
        except Exception:
            errors.report(f"Error reading webui git info from {script_path}", exc_info=True)

    # Extract repository information if available
    if webui_repo and not webui_repo.bare:
        try:
            webui_remote = next(webui_repo.remote().urls, None)
            head_commit = webui_repo.head.commit
            webui_commit_date = head_commit.committed_date
            webui_commit_hash = head_commit.hexsha
            webui_branch = webui_repo.active_branch.name
        except Exception:
            # If something goes wrong, just leave these as None
            webui_remote = None

    return {
        "remote": webui_remote,
        "commit_hash": webui_commit_hash,
        "commit_date": webui_commit_date,
        "branch": webui_branch,
    }


def get_extension_config():
    """
    Gathers configuration data for all extensions.

    Returns:
        dict: A dictionary keyed by extension name, each value containing details:
              - name, path, enabled, is_builtin, remote, commit_hash,
                commit_date, branch, have_info_from_repo
    """
    ext_config = {}

    for ext in extensions.extensions:
        # Update the extension info from its repository
        ext.read_info_from_repo()

        ext_config[ext.name] = {
            "name": ext.name,
            "path": ext.path,
            "enabled": ext.enabled,
            "is_builtin": ext.is_builtin,
            "remote": ext.remote,
            "commit_hash": ext.commit_hash,
            "commit_date": ext.commit_date,
            "branch": ext.branch,
            "have_info_from_repo": ext.have_info_from_repo
        }

    return ext_config


def get_config():
    """
    Retrieves the current configuration state of the WebUI and its extensions.

    Returns:
        dict: A dictionary containing the 'webui' and 'extensions' keys.
    """
    creation_time = datetime.now().timestamp()
    webui_config = get_webui_config()
    extension_config = get_extension_config()

    return {
        "created_at": creation_time,
        "webui": webui_config,
        "extensions": extension_config
    }


def reset_to_commit(repo, commit_hash):
    """Helper to fetch all branches and reset the repo to a given commit."""
    repo.git.fetch(all=True)
    repo.git.reset(commit_hash, hard=True)


def restore_webui_config(config):
    """
    Restore the WebUI repository to a previously saved commit.

    Args:
        config (dict): The configuration state dictionary that includes 'webui' section.
    """
    print("* Restoring webui state...")

    if "webui" not in config:
        print("Error: No WebUI data found in this configuration.")
        return

    webui_config = config["webui"]
    webui_commit_hash = webui_config.get("commit_hash")

    if not webui_commit_hash:
        print("Error: No commit hash found in the WebUI configuration.")
        return

    webui_repo = None
    if os.path.exists(os.path.join(script_path, ".git")):
        try:
            webui_repo = git.Repo(script_path)
        except Exception:
            errors.report(f"Error reading WebUI git info from {script_path}", exc_info=True)
            return
    else:
        print("Error: The WebUI directory is not a git repository.")
        return

    # Attempt to restore the WebUI to the saved commit
    try:
        reset_to_commit(webui_repo, webui_commit_hash)
        print(f"* Restored WebUI to commit {webui_commit_hash}.")
    except Exception:
        errors.report(f"Error restoring WebUI to commit {webui_commit_hash}", exc_info=True)


def restore_extension_config(config):
    """
    Restore all extensions to a previously saved state (including their commit and enabled/disabled status).

    Args:
        config (dict): The configuration state dictionary that includes 'extensions' section.
    """
    print("* Restoring extension state...")

    if "extensions" not in config:
        print("Error: No extension data found in this configuration.")
        return

    saved_extension_config = config["extensions"]

    # Keep track of the outcomes for each extension
    results = []
    disabled_extensions = []

    # Progress bar to show extension restoration progress
    for ext in tqdm(extensions.extensions, desc="Restoring extensions"):
        if ext.is_builtin:
            # Skip built-in extensions (no separate repo to revert)
            continue

        # Refresh extension info from its repo to get current commit
        ext.read_info_from_repo()
        current_commit = ext.commit_hash
        saved_entry = saved_extension_config.get(ext.name)

        if saved_entry is None:
            # If the extension doesn't exist in the saved config, disable it
            ext.disabled = True
            disabled_extensions.append(ext.name)
            results.append((ext, current_commit[:8] if current_commit else "None", False, "Not found in saved config, disabled"))
            continue

        # Attempt to restore to the saved commit hash (if one exists)
        target_commit = saved_entry.get("commit_hash")
        if target_commit:
            try:
                ext.fetch_and_reset_hard(target_commit)
                ext.read_info_from_repo()
                # Check if the commit actually changed
                if current_commit != target_commit:
                    results.append((ext, current_commit[:8] if current_commit else "None", True, target_commit[:8]))
            except Exception as ex:
                results.append((ext, current_commit[:8] if current_commit else "None", False, str(ex)))
        else:
            # No commit hash in the saved config means we can't restore it
            results.append((ext, current_commit[:8] if current_commit else "None", False, "No commit hash in saved config"))

        # Update extension enabled/disabled state
        if not saved_entry.get("enabled", False):
            ext.disabled = True
            disabled_extensions.append(ext.name)
        else:
            ext.disabled = False

    # Update and save global configuration with new disabled extensions
    shared.opts.disabled_extensions = disabled_extensions
    shared.opts.save(shared.config_filename)

    # Print a summary of the extension restore results
    print("* Finished restoring extensions. Results:")
    for ext, prev_commit, success, info in results:
        if success:
            print(f"  + {ext.name}: {prev_commit} -> {info}")
        else:
            print(f"  ! {ext.name}: FAILURE ({info})")

#  what other features can we add TODO