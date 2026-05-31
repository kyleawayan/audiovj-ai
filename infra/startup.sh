#!/usr/bin/env bash
#
# AudioVJ GPU runner — boot-time provisioning (runs as root via the GCP metadata
# startup-script mechanism). System-level prep only; per-user steps (uv, repo
# clone, CUDA torch) live in /opt/audiovj/setup.sh which the user runs once.
#
# Contains NO secrets. Claude Code auth is done interactively by the user.

set -uo pipefail
exec > >(tee -a /var/log/audiovj-startup.log) 2>&1
echo "[audiovj] startup begin $(date -u)"

md() { # read an instance metadata attribute
  curl -s -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" 2>/dev/null
}

BUCKET="$(md bucket-name)"
ARCHIVE="$(md archive-object)"
REPO_BUNDLE="$(md repo-bundle)"
IDLE_MIN="$(md idle-shutdown-minutes)"
: "${IDLE_MIN:=0}"

#######################################
# Packages
#######################################
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || true
apt-get install -y tmux git curl gnupg lsb-release fuse ca-certificates || true

# gcsfuse (Cloud Storage FUSE)
if ! command -v gcsfuse >/dev/null 2>&1; then
  echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" \
    > /etc/apt/sources.list.d/gcsfuse.list
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null || true
  apt-get update -y || true
  apt-get install -y gcsfuse || true
fi

# Let the (non-root) OS Login user access the FUSE mount.
grep -q '^user_allow_other' /etc/fuse.conf 2>/dev/null || echo 'user_allow_other' >> /etc/fuse.conf

#######################################
# Mount the GCS bucket at /mnt/bucket (durable; code-in / results-out)
#######################################
mkdir -p /mnt/bucket
cat > /etc/systemd/system/gcsfuse-bucket.service <<EOF
[Unit]
Description=gcsfuse mount of ${BUCKET}
After=network-online.target
Wants=network-online.target
[Service]
Type=simple
ExecStart=/usr/bin/gcsfuse --foreground -o allow_other --implicit-dirs --file-mode=664 --dir-mode=775 ${BUCKET} /mnt/bucket
ExecStop=/bin/fusermount -u /mnt/bucket
Restart=on-failure
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable --now gcsfuse-bucket.service || true

#######################################
# Local NVMe SSD scratch at /mnt/scratch (fast, ephemeral)
#######################################
mkdir -p /mnt/scratch
SSD=/dev/disk/by-id/google-local-nvme-ssd-0
if [ -e "$SSD" ]; then
  if ! blkid "$SSD" >/dev/null 2>&1; then
    echo "[audiovj] formatting local SSD $SSD"
    mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "$SSD"
  fi
  if ! mountpoint -q /mnt/scratch; then
    mount -o discard,defaults "$SSD" /mnt/scratch
  fi
  chmod 1777 /mnt/scratch
else
  echo "[audiovj] WARN: no local SSD found at $SSD (local_ssd_count=0?)"
fi

#######################################
# Stream-extract the data archive from the bucket onto the SSD (idempotent)
#######################################
if [ -n "$BUCKET" ] && [ -n "$ARCHIVE" ] && [ ! -d /mnt/scratch/data/features ]; then
  echo "[audiovj] extracting gs://${BUCKET}/${ARCHIVE} -> /mnt/scratch"
  if gcloud storage cat "gs://${BUCKET}/${ARCHIVE}" | tar -x -C /mnt/scratch; then
    chmod -R a+rX /mnt/scratch/data || true
    echo "[audiovj] archive extracted"
  else
    echo "[audiovj] WARN: extract failed. Upload gs://${BUCKET}/${ARCHIVE}, then run: sudo bash /opt/audiovj/extract-archive.sh"
  fi
else
  echo "[audiovj] archive already present or not configured; skipping extract"
fi

#######################################
# Claude Code settings -> deployed to new user homes via /etc/skel
#######################################
mkdir -p /etc/skel/.claude /opt/audiovj
md claude-settings > /etc/skel/.claude/settings.json
cp -f /etc/skel/.claude/settings.json /opt/audiovj/claude-settings.json
chmod 0644 /opt/audiovj/claude-settings.json

#######################################
# Helper scripts (user-run)
#######################################

# Re-extract the archive on demand (e.g. after a stop wiped the local SSD).
cat > /opt/audiovj/extract-archive.sh <<'XEOF'
#!/usr/bin/env bash
set -euo pipefail
md() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
BUCKET="$(md bucket-name)"; ARCHIVE="$(md archive-object)"
echo "Extracting gs://${BUCKET}/${ARCHIVE} -> /mnt/scratch ..."
gcloud storage cat "gs://${BUCKET}/${ARCHIVE}" | sudo tar -x -C /mnt/scratch
sudo chmod -R a+rX /mnt/scratch/data
echo "Done."
XEOF
chmod 0755 /opt/audiovj/extract-archive.sh

# One-time per-user setup: uv, repo clone from the bucket bundle, CUDA torch.
cat > /opt/audiovj/setup.sh <<'SEOF'
#!/usr/bin/env bash
set -euo pipefail
md() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
REPO_BUNDLE="$(md repo-bundle)"
REPO="$HOME/audiovj-ai"

# Ensure ~/.claude/settings.json exists for THIS user (in case /etc/skel didn't apply).
mkdir -p "$HOME/.claude"
[ -f "$HOME/.claude/settings.json" ] || cp /opt/audiovj/claude-settings.json "$HOME/.claude/settings.json"

# uv
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Clone the repo from the git bundle on the bucket (no GitHub login needed).
if [ ! -d "$REPO/.git" ]; then
  git clone "/mnt/bucket/${REPO_BUNDLE}" "$REPO"
fi
cd "$REPO"

# Point data/ at the local SSD scratch.
mkdir -p /mnt/scratch/data
ln -sfn /mnt/scratch/data "$REPO/data"

# Python deps, then replace CPU torch with the CUDA build.
uv sync
uv pip install --reinstall --index-url https://download.pytorch.org/whl/cu124 torch torchaudio

echo
echo "Setup complete. Verify GPU:    uv run python -c 'import audiovj.training as t; print(t._get_device())'"
echo "Then:  claude   (log in with your Claude subscription)"
echo "Then:  tmux new -s loop  &&  cd $REPO  &&  claude"
echo "Export results to bucket:  bash /opt/audiovj/export-results.sh"
SEOF
chmod 0755 /opt/audiovj/setup.sh

# Push commits + checkpoints back to the bucket (no GitHub from the VM).
cat > /opt/audiovj/export-results.sh <<'EEOF'
#!/usr/bin/env bash
set -euo pipefail
REPO="${1:-$HOME/audiovj-ai}"
TS="$(date -u +%Y%m%d-%H%M%S)"
mkdir -p /mnt/bucket/out
cd "$REPO"
git bundle create "/mnt/bucket/out/repo-${TS}.bundle" --all
if compgen -G "data/models/*.safetensors" > /dev/null; then
  mkdir -p "/mnt/bucket/out/models-${TS}"
  cp -f data/models/*.safetensors "/mnt/bucket/out/models-${TS}/"
fi
cp -f ./*.log "/mnt/bucket/out/" 2>/dev/null || true
echo "Exported repo-${TS}.bundle + checkpoints/logs to the bucket (/mnt/bucket/out/)."
echo "On your Mac:  gcloud storage cp -r gs://<bucket>/out ./out ; git fetch out/repo-${TS}.bundle '*:*'"
EEOF
chmod 0755 /opt/audiovj/export-results.sh

#######################################
# Idle auto-shutdown (CPU + GPU both low for IDLE_MIN minutes)
#######################################
if [ "${IDLE_MIN}" -gt 0 ] 2>/dev/null; then
  cat > /opt/audiovj/idle-check.sh <<'IEOF'
#!/usr/bin/env bash
# Shut down if GPU util < 10% AND 1-min load < 1.0 for >= IDLE_MIN minutes.
set -uo pipefail
md() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
IDLE_MIN="$(md idle-shutdown-minutes)"; : "${IDLE_MIN:=0}"
[ "$IDLE_MIN" -gt 0 ] || exit 0
INTERVAL=5
STATE=/var/run/audiovj-idle-min
gpu="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
: "${gpu:=0}"
load="$(awk '{print $1}' /proc/loadavg)"
busy=0
[ "${gpu}" -ge 10 ] 2>/dev/null && busy=1
awk "BEGIN{exit !(${load} >= 1.0)}" && busy=1
if [ "$busy" -eq 1 ]; then echo 0 > "$STATE"; exit 0; fi
n="$(cat "$STATE" 2>/dev/null || echo 0)"; n=$((n + INTERVAL)); echo "$n" > "$STATE"
if [ "$n" -ge "$IDLE_MIN" ]; then
  logger -t audiovj "idle ${n}m >= ${IDLE_MIN}m — shutting down"
  /sbin/shutdown -h now
fi
IEOF
  chmod 0755 /opt/audiovj/idle-check.sh

  cat > /etc/systemd/system/audiovj-idle.service <<'EOF'
[Unit]
Description=AudioVJ idle shutdown check
[Service]
Type=oneshot
ExecStart=/opt/audiovj/idle-check.sh
EOF
  cat > /etc/systemd/system/audiovj-idle.timer <<'EOF'
[Unit]
Description=Run AudioVJ idle check every 5 minutes
[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
[Install]
WantedBy=timers.target
EOF
  systemctl daemon-reload
  systemctl enable --now audiovj-idle.timer || true
fi

echo "[audiovj] startup done $(date -u)"
