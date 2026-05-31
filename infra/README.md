# AudioVJ GPU training sandbox (GCP / Terraform)

An **isolated, disposable** GCP GPU VM for running a Claude Code feedback loop on the
AudioVJ phrase-detection model, plus a private GCS bucket to move data and code in/out.

## What it builds

- **New-project isolation** — only `compute`, `storage`, `iap`, `iam` APIs are enabled; every
  other GCP API stays off.
- **GPU VM** `n1-highmem-8` + **1× T4** + **375GB local NVMe SSD**. The workload is
  data-pipeline-bound (tiny model, ~47GB of mmap'd features read every epoch), so the RAM and
  fast local disk matter more than the GPU. Flip to L4/spot/etc. via variables.
- **Private GCS bucket** (uniform access, versioning, public access blocked).
- **`runner-sa`** — the VM's identity, with **zero project roles**; its only access is
  `objectAdmin` on the one bucket, and instance OAuth scopes are capped to storage + logging.
  The VM literally cannot reach other GCP services.
- **`transfer-sa`** — a bucket-only identity you **impersonate** from your laptop for keyless
  uploads (no JSON key files).
- **IAP-only access** (no public IP) + **Cloud NAT** for outbound internet (Anthropic API,
  PyPI, apt). Optional **hardened egress** allow-list.
- **Idle auto-shutdown** and an optional **billing budget**.

## Prerequisites (one-time, on your Mac)

1. Create a **brand-new GCP project** and link billing. (Kept manual so Terraform never needs
   org/billing-admin rights. A project-creation module can be added if you want.)
2. `gcloud auth login` and `gcloud auth application-default login`.
3. `gcloud config set project <PROJECT_ID>`.
4. Install Terraform (`>= 1.5`).

## Deploy

```bash
cd infra
cp terraform.tfvars.example terraform.tfvars   # then edit: project_id, bucket_name, iap_user
terraform init
terraform fmt && terraform validate
terraform plan      # review: no public IP, runner-sa has no project roles, only 4 APIs
terraform apply
```

`terraform output` prints the browser-SSH URL and ready-to-paste upload commands.

## Get data + code onto the box (no GitHub on the VM)

All transfer is keyless (your OAuth login impersonating `transfer-sa`), bucket-only:

```bash
# 1) data archive (your existing ~150GB tar — transfer it whole, no subsetting)
gcloud storage cp <your-archive>.tar gs://<bucket>/archive.tar \
  --impersonate-service-account=<transfer-sa-email>

# 2) code as a git bundle (run inside your local repo on the ka-234 branch)
git bundle create /tmp/repo.bundle --all
gcloud storage cp /tmp/repo.bundle gs://<bucket>/repo.bundle \
  --impersonate-service-account=<transfer-sa-email>
```

`terraform output upload_archive_command` / `upload_repo_bundle_command` give the exact lines.

## Drive the loop

1. Open the **browser-SSH** URL from `terraform output browser_ssh_url` (tunnels via IAP — no
   local SSH client or keys).
2. The startup script has already: mounted the bucket at `/mnt/bucket`, extracted the archive
   to `/mnt/scratch`, installed gcsfuse/tmux, and dropped the Claude settings into `/etc/skel`.
   If you uploaded the archive *after* boot, run `sudo bash /opt/audiovj/extract-archive.sh`.
3. One-time per-user setup:
   ```bash
   bash /opt/audiovj/setup.sh      # uv, clone repo from /mnt/bucket/repo.bundle, CUDA torch, symlink data/
   uv run python -c 'import audiovj.training as t; print(t._get_device())'   # -> cuda
   ```
4. Auth + run:
   ```bash
   claude                 # log in with your Claude subscription (OAuth; no API key)
   tmux new -s loop
   cd ~/audiovj-ai && claude     # steer this from claude.ai / phone via Remote Control
   ```
   Claude Code runs in **auto mode** with a single hard rule: it can never read
   `~/.claude/.credentials.json` (see `claude-settings.json`).

## Get results back

Claude commits **locally** on the VM (no GitHub). To pull them down:

```bash
# on the VM
bash /opt/audiovj/export-results.sh        # writes repo-<ts>.bundle + checkpoints/logs to /mnt/bucket/out/

# on your Mac
gcloud storage cp -r gs://<bucket>/out ./out --impersonate-service-account=<transfer-sa-email>
cd <your-local-repo>
git fetch ./out/repo-<ts>.bundle '*:refs/remotes/vm/*'   # inspect the VM's commits
cp ./out/models-<ts>/*.safetensors data/models/          # for local inference
```

## Cost & teardown

≈ **$0.87/hr** on-demand (`n1-highmem-8` + T4 + local SSD) → ~115 hrs in $100. The bucket
persists at ~$5/mo. To stop spend:

- `idle_shutdown_minutes` stops the VM automatically when idle (local SSD scratch is wiped;
  re-extract on next boot — it's automatic, or `extract-archive.sh`).
- `terraform destroy` removes the VM (and local SSD). The bucket survives unless
  `bucket_force_destroy = true`.

## Notes / knobs

- **Security model:** the VM is the boundary. `runner-sa` is bucket-only, there are no GitHub
  creds on the box, and `git push` can't leak anywhere. The Claude OAuth token lives only as a
  `0600` file on this throwaway VM and is revocable from your Claude account. File mode does
  **not** stop Claude itself from reading its own token — if you want that airtight, set
  `harden_egress = true` so nothing can be sent to an arbitrary host.
- **`local_ssd_count > 1`** needs a software RAID across the NVMe devices — the single-disk
  startup logic doesn't handle that; add an `mdadm` step if you raise it.
- **GPU swap:** `nvidia-l4` requires a `g2-*` machine type (not `n1-*`); change both together.
- Verify `"defaultMode": "auto"` matches your installed Claude Code version's mode name.
