# Hermes agent sync

Keep Hermes agents in sync by sharing their profile's instructions, preferences, and skills through a private GitHub repository.

> [!NOTE]
> The plugin synchronizes selected files from each agent’s profile, sessions are not included.<br/>
> Profiles can belong to the same gateway or different gateways, cf. [Terminology](#terminology) and [Setup](#setup).

## What is synced

File selection is configured by [`sync.yaml`](#1-setup-data-repository-with-syncyaml
). The default selection includes:

| File | Description |
| --- | --- |
| `SOUL.md` | Agent instructions |
| `memories/USER.md` | User preferences |
| `skills/**` | Skills and supporting files, including newly added skills |

<br/>

> [!TIP]
> Machine notes (`memories/MEMORY.md`), credentials, sessions, and caches should not be synced.

## Getting started

### 1. Setup data repository with `sync.yaml`

Synced files are stored in a git repository that should be kept private. Ex. on GitHub:

```bash
gh repo create https://github.com/ACCOUNT/hermes-data.git --private
```

For each profile to sync, create a branch containing `sync.yaml` (at root), listing what to sync:

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

### 2. Install / config

On each Hermes profile to sync, install the plugin:

```bash
hermes -p PROFILE plugins install capsudo/hermes-agent-sync --enable
```

Then set the data repo URL to use and set up the branch name to sync with (see [Setup](#setup) which explains this):

```bash
hermes -p PROFILE config set plugins.entries.profile-sync.settings.data-repo-url https://github.com/ACCOUNT/hermes-data.git
hermes -p PROFILE config set plugins.entries.profile-sync.settings.data-repo-branch PROFILE
```

> [!TIP]
> If using Hermes Desktop those fields can be configured under Settings → Plugins → Profile Sync.

### 3. Trigger the sync

Actual syncing of selected files can be done in two ways:

- **Manual sync**: ask Hermes agent (in chat).
- **Scheduled checks** (automatic synchronization at a configured interval): not implemented yet.

> [!NOTE]
> First sync combines existing content without treating a missing file as a deletion.

## Setup

### Single data repository

The default setup is to have a single data repository (`data-repo-url`) where each branch represents a profile.

#### Sync profile across different gateways

| Gateway | Profile | Data branch |
| --- | --- | --- |
| Local | `default` | `default` |
| Remote | `default` | `default` |

> [!TIP]
> This is useful for running the same agent on different gateways, for instance an agent for Computer Use on a **Local** gateway and another for long running tasks on a **Remote** gateway, cf. [Terminology](#terminology).

#### Sync profiles within a gateway

Two profiles in the same gateway can also use the same branch. They sync selected content just like profiles on different gateways, for instance:

| Gateway | Profile | Data branch |
| --- | --- | --- |
| Local | `default` | `default` |
| Local | `special` | `default` |

> [!TIP]
> This allows sessions to be split into different profiles while sharing skills and preferences, or anything that is listed for sync, cf. [Terminology](#terminology).

#### Sync multiple profiles

On each profile, `data-repo-branch` is used to select the branch to sync with.<br/>
Usually branch name = profile name but they can differ, for instance:

| Gateway | Profile | Data branch |
| --- | --- | --- |
| Local | `default` | `default` |
| Local | `work` | `work` |
| Remote | `default` | `default` |
| Remote | `research` | `work` |

> [!NOTE]
> Each branch contains its own `sync.yaml` (at root) listing selected files for syncing.<br/>
> Creating another profile does not enroll it automatically, it's opt-in.

### Multiple data repositories

Using different data repository URLs for different profiles is also possible:

| Gateway | Profile | Data Repository | Data branch |
| --- | --- | --- | --- |
| Local | `default` | github.com/personal/hermes-data.git | `default` |
| Remote | `default` | github.com/personal/hermes-data.git | `default` |
| Local | `work` | github.com/organization/hermes-data.git | `default` |
| Remote | `work` | github.com/organization/hermes-data.git | `default` |

## How it works

### Workflow

Script does git stuff and checks, LLM does the merge and resolves potential conflicts:

1. Fetch latest changes from data repository and snapshot selected files from local profile.
2. Compare current and remote versions, skip (stop here) if they match.
3. Compare both against the previous synced version to identify where the change comes from.
4. LLM checks every actual change, including apparently straightforward merges.
5. Apply the result locally; automatically if straightforward, otherwise propose a resolution and ask a short question to decide if needed.
6. Commit/push changes to the data repository (only if result from step 5 differs from the repo version).
7. Update local ref to "last sync" commit.

> [!WARNING]
> LLM review sends relevant file content to the model configured in Hermes.

### Local vs repo changes

Comparing **local** and **repo** files shows differences, but not which side changed.
For example, a line present only locally could have been added locally or deleted in repo.

Instead of relying on modification date, 3 versions are compared:

- repo version: file in data repository.
- local current version: file on disk.
- local previous version: file at **last sync** commit for _this profile_ on _this gateway_

> [!NOTE]
> A local Git reference (`refs/profile-sync/base`) is used to mark this last successful sync commit.

## Terminology

### Agent

Running system that uses a model (LLM) and tools to answer messages and perform tasks. Its profile supplies instructions, preferences, skills, and settings.

### Profile

Set of sessions, instructions, preferences, and skills kept separate for different uses, such as personal and work.

### Gateway

Hermes backend serving profiles and where the agents run.<br/>
**Local** gateway means running on a workstation and a **Remote** gateway means running on Hermes Cloud or a VPS.