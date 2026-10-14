import base64
import os
from pathlib import Path


def push_run(files, message, branch="training"):
    # Commits the given files to GitHub in one commit through Composio.
    from composio import Composio

    owner = os.environ["GITHUB_OWNER"]
    repo = os.environ.get("GITHUB_REPO", "fly-brain-racing")
    upserts = []
    for rel in files:
        p = Path(rel)
        if p.exists():
            upserts.append({"path": rel, "content": base64.b64encode(p.read_bytes()).decode(), "encoding": "base64"})
    if not upserts:
        return None

    arguments = {"owner": owner, "repo": repo, "branch": branch, "base_branch": "main",
                 "message": message, "upserts": upserts}
    client = Composio(api_key=os.environ["COMPOSIO_API_KEY"])
    kwargs = {"user_id": os.environ.get("COMPOSIO_USER_ID", "default")}
    if os.environ.get("COMPOSIO_CONNECTED_ACCOUNT_ID"):
        kwargs["connected_account_id"] = os.environ["COMPOSIO_CONNECTED_ACCOUNT_ID"]
    return client.tools.execute("GITHUB_COMMIT_MULTIPLE_FILES", arguments,
                                dangerously_skip_version_check=True, **kwargs)
