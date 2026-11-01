# AML compute operations

This directory contains the tools used to run Kuno training on Azure Machine Learning compute nodes. The controller finds an available node, prepares storage and services, starts training when enabled, and reports progress.

## Main files

| File | Purpose |
| --- | --- |
| `amlrunner.py` | Finds a node, checks its health, repairs it when needed, and monitors training. |
| `remote/bootstrap.sh` | Installs and configures storage and training services on the node. |
| `remote/run-training.sh` | Checks the node and starts the training container. |

## Setup

You need:

- Azure CLI signed in with access to AML, the storage account, and the container registry.
- Python 3.10 or newer with `training/amlcompute/requirements.txt` installed.
- A `.env` file based on `.env.example`.
- An SSH user that can run `sudo`.

Set `AML_COMPUTE_RESOURCE_IDS` to a comma-separated list of AML compute resource IDs. The runner checks them in the order listed. All computes must use the same SSH credentials.

AML may reimage a node while retaining its public endpoint. When the host key changes for a node returned by the authenticated ARM discovery API, the runner removes only that endpoint's stale key and retries once, then saves the replacement key.

Set `KUNO_IMAGE` to an immutable image digest. If the image may not already exist on a replacement node, configure `ACR_USERNAME` and `ACR_PASSWORD`; the runner uses them through a temporary Docker configuration to pull the image and removes that configuration before starting the service.

## Run

Start the controller and keep it running:

    python training/amlcompute/amlrunner.py run

Check and repair once, then exit:

    python training/amlcompute/amlrunner.py run --once

Change the check interval:

    python training/amlcompute/amlrunner.py run --interval 60

Stop training and unmount storage on the selected node:

    python training/amlcompute/amlrunner.py stop

## How node selection works

The runner remembers the selected compute and node in `.state/selected-node.json`. It reuses that node when it still belongs to the same training run. Otherwise, it chooses the first Idle node from the configured computes. It will not take a busy node that may belong to someone else.

The logs include both the compute name and node ID. Logs are written to the terminal and to `training/amlcompute/.state/amlrunner.log`.

## Storage and credentials

Training data is mounted with BlobFuse2 at `KUNO_DURABLE_PATH` (`/mnt/kuno-training` by default), and node-local scratch lives at `KUNO_LOCAL_PATH` (`/mnt/kuno-local` by default). Both directories are created by `bootstrap.sh` and owned by the service user. The runner checks the mount before training starts and repairs it when needed.

The storage account key is kept in `/etc/kuno-storage.env` on the node with permissions limited to root. It is not passed to the training container or written to the logs.

ACR credentials remain in the controller's local `.env`. They are not copied into `/etc/kuno-training.env` or retained on the node after an image pull.

## Training behavior

Set `KUNO_TRAINING_ENABLED=true` to run real training. When it is `false` or not set, the runner performs a storage dry run instead.

`KUNO_RUN_ID` is required for real training. A dry run uses `dry-run` when no run ID is set.

Temporary failures restart the same run. Fatal failures create `runs/<run-id>/HALT.json` and stop restarting. Run state is reported in `runs/<run-id>/status.json`, which records `running`, `completed`, `retryable`, or `fatal`.
