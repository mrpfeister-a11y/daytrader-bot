"""Email Miles the same way the other bots do: open a GitHub issue in this repo that
@mentions him, then close it. GitHub emails it to him. No passwords, no extra app.

Note: this repo is public, so these issues are public too. They only contain
paper-money numbers - never keys or account numbers.
"""
import json
import os
import urllib.request


def _api(path, method="GET", body=None):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if not (token and repo):
        return None
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/{path}", method=method,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "Content-Type": "application/json"},
        data=json.dumps(body).encode() if body is not None else None)
    with urllib.request.urlopen(req, timeout=20) as r:
        txt = r.read()
        return json.loads(txt) if txt else {}


def send(title: str, message: str, urgent: bool = False) -> bool:
    """Never raises - an email problem must not break trading."""
    title = ("⚠️ " if urgent else "") + "Day-trader: " + title
    who = os.environ.get("NOTIFY_MENTION", "").strip()
    body = message + (f"\n\n@{who}" if who else "") + \
        "\n\n_Sent automatically by your day-trading bot (paper money - not financial advice)._"
    try:
        issue = _api("issues", "POST", {"title": title, "body": body[:65000]})
        if issue is None:
            print(f"[email not sent - not running on GitHub]\n{title}\n{message}")
            return False
        _api(f"issues/{issue['number']}", "PATCH", {"state": "closed"})
        print(f"email sent (issue #{issue['number']}): {title}")
        return True
    except Exception as e:
        print(f"email failed (ignored): {e}")
        return False


def start_run(mode: str, workflow: str = "bot.yml") -> bool:
    """Start another run of a workflow (bot.yml = bot #8, sip.yml = bot #9). Used to hand
    over a long trading day to a fresh job, to restart after a crash, and to retrain."""
    try:
        r = _api(f"actions/workflows/{workflow}/dispatches", "POST",
                 {"ref": "main", "inputs": {"mode": mode}})
        print(f"started a new '{mode}' run" if r is not None else "[not on GitHub - no new run]")
        return r is not None
    except Exception as e:
        print(f"could not start a new run: {e}")
        return False
