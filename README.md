# Hermes profile sync

Share instructions, preferences, and skills between different Hermes agents, through a private GitHub repository.

> This is useful when using agents on different machines, for instance a local agent for Computer Use and a remote one on Hermes Cloud for long running tasks.

## What is synced

File selection is configured by `sync.yaml` in the [data repository](#data-repository).
The default selection includes:

| File | Description |
| --- | --- |
| `SOUL.md` | Agent instructions |
| `memories/USER.md` | User preferences |
| `skills/**` | Skills and supporting files, including newly added skills |

<br/>

> [!TIP]
> Machine notes (`memories/MEMORY.md`), credentials, sessions, and caches should not be synced.


## Data repository

Synced files are stored in a (private) GitHub repository.
It must contain `sync.yaml` at root, listing what to sync:

```yaml
version: 1
skip-symlinks: true
include:
  - SOUL.md
  - memories/USER.md
  - skills/**
exclude: [] # Optional profile-relative exclusion patterns
```

> [!NOTE]
> Symlinks and excluded files are skipped, not treated as deleted.<br/>
> Built-in safety rules also exclude credentials and runtime state.<br/>
> Exclusion patterns override inclusion patterns.

## Installation / Config

Install the plugin then define data repo URL + branch on each Hermes profile to sync:

```bash
hermes plugins install capsudo/hermes-profile-sync --enable
hermes config set plugins.entries.profile-sync.settings.data-repo-url https://github.com/ACCOUNT/hermes-profile.git
hermes config set plugins.entries.profile-sync.settings.data-repo-branch master
```

> [!TIP]
> If using Hermes Desktop those fields can be configured under Settings → Plugins → Profile Sync

## Syncing

Actual syncing of selected files can be done in two ways:

- **Manual sync**: ask Hermes agent (in chat).
- **Scheduled checks** (automatic synchronization at a configured interval): not implemented yet.

> [!NOTE]
> First sync combines existing content without treating a missing file as a deletion.

### Determine file change origin

Comparing local and remote files shows differences, but not which side changed.
For example, a line present only locally could have been added locally or deleted remotely.

Instead of relying on modification date, 3 versions are compared:

- remote version: file in data repository.
- current version: file in local profile.
- previous synced version: file at **last sync** commit (last _successful_ commit on _this machine_)

> [!NOTE]
> A local Git reference, `refs/profile-sync/base`, points to the commit containing the previous synced version.

### Workflow

LLM does the merge and resolves potential conflicts:

1. Fetch latest changes from data repository and snapshot selected files from local profile.
2. Compare current and remote versions, skip (stop here) if they match.
3. Compare both against the previous synced version to identify where the change comes from.
4. LLM checks every actual change, including apparently straightforward merges.
5. Apply the result locally; automatically if straightforward, otherwise propose a resolution and ask a short question to decide if needed.
6. Commit changes to the data repository and push. Update local ref to this "last sync" commit.

> [!WARNING]
> LLM review sends relevant file content to the model configured in Hermes.
