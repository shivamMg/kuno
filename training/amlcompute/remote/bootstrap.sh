#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: bootstrap.sh AUTH_ENV_FILE NODE_ID USER MOUNT_PATH LOCAL_PATH" >&2
  exit 2
fi

AUTH_ENV_FILE="$1"
NODE_ID="$2"
SERVICE_USER="$3"
MOUNT_PATH="$4"
LOCAL_PATH="$5"
CACHE_PATH="/var/cache/blobfuse2/kuno-training"
TRAINING_ENV_FILE="/tmp/kuno-training.env"
TRAINING_SCRIPT="/tmp/run-training.sh"
TRAINING_SERVICE="/tmp/kuno-training.service"

# Never leave the storage account key or runtime settings behind in /tmp,
# including on the early-exit validation paths below.
trap 'rm -f "$AUTH_ENV_FILE" "$TRAINING_ENV_FILE" "$TRAINING_SCRIPT" "$TRAINING_SERVICE"' EXIT

if [[ ! -f "$AUTH_ENV_FILE" ]]; then
  echo "Storage authentication environment file is missing" >&2
  exit 1
fi
for required_file in "$TRAINING_ENV_FILE" "$TRAINING_SCRIPT" "$TRAINING_SERVICE"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing training service input: $required_file" >&2
    exit 1
  fi
done
install -o root -g root -m 0600 "$AUTH_ENV_FILE" /etc/kuno-storage.env
rm -f "$AUTH_ENV_FILE"
install -o root -g root -m 0600 "$TRAINING_ENV_FILE" /etc/kuno-training.env
sed -i 's/\r$//' "$TRAINING_SCRIPT"
install -o root -g root -m 0755 "$TRAINING_SCRIPT" /usr/local/bin/run-training.sh
sed \
  -e "s|__SERVICE_USER__|$SERVICE_USER|g" \
  -e "s|__MAX_RESTARTS__|5|g" \
  "$TRAINING_SERVICE" >/etc/systemd/system/kuno-training.service
chmod 0644 /etc/systemd/system/kuno-training.service
rm -f "$TRAINING_ENV_FILE" "$TRAINING_SCRIPT" "$TRAINING_SERVICE"

if ! command -v blobfuse2 >/dev/null 2>&1; then
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    echo "Automatic BlobFuse2 installation currently supports Ubuntu; found ${ID:-unknown}" >&2
    exit 1
  fi
  apt-get update
  apt-get install -y ca-certificates curl fuse3
  curl -fsSL "https://packages.microsoft.com/config/ubuntu/${VERSION_ID}/packages-microsoft-prod.deb" \
    -o /tmp/packages-microsoft-prod.deb
  dpkg -i /tmp/packages-microsoft-prod.deb
  apt-get update
  apt-get install -y blobfuse2
fi

install -d -o "$SERVICE_USER" -g "$SERVICE_USER" -m 0755 "$MOUNT_PATH" "$CACHE_PATH" "$LOCAL_PATH"
grep -qxF user_allow_other /etc/fuse.conf || echo user_allow_other >> /etc/fuse.conf

# Remove the retired heartbeat service before remounting storage.
systemctl disable --now kuno-sanity.service 2>/dev/null || true
rm -f /etc/systemd/system/kuno-sanity.service /usr/local/bin/kuno-sanity-heartbeat
# Stop the training unit and reap any container that outlived it; otherwise the
# container keeps the bind mount busy and the unmount below fails with EBUSY.
systemctl stop kuno-training.service 2>/dev/null || true
docker rm -f kuno-training >/dev/null 2>&1 || true
systemctl stop kuno-blobfuse2.service 2>/dev/null || true
if mountpoint -q "$MOUNT_PATH"; then
  fusermount3 -u "$MOUNT_PATH"
fi
rm -rf "$MOUNT_PATH/sanity"
if find "$MOUNT_PATH" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "Refusing to mount over non-empty local directory: $MOUNT_PATH" >&2
  exit 1
fi

cat >/etc/systemd/system/kuno-blobfuse2.service <<EOF
[Unit]
Description=Kuno training Azure Blob Storage mount
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$SERVICE_USER
EnvironmentFile=/etc/kuno-storage.env
ExecStart=/usr/bin/blobfuse2 mount $MOUNT_PATH --tmp-path=$CACHE_PATH --allow-other --foreground=true --log-type=syslog --log-level=LOG_WARNING
ExecStop=/bin/fusermount3 -u $MOUNT_PATH
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable kuno-blobfuse2.service kuno-training.service
systemctl start kuno-blobfuse2.service

for _ in {1..30}; do
  mountpoint -q "$MOUNT_PATH" && break
  sleep 1
done
mountpoint -q "$MOUNT_PATH"
probe_dir="$MOUNT_PATH/.kuno-probes"
probe_path="$probe_dir/bootstrap-${NODE_ID}-$$"
probe_content="node=$NODE_ID status=ok"
install -d -m 0755 "$probe_dir"
mountpoint -q "$MOUNT_PATH"
printf '%s\n' "$probe_content" >"$probe_path"
mountpoint -q "$MOUNT_PATH"
[[ "$(cat "$probe_path")" == "$probe_content" ]]
rm -f "$probe_path"
rmdir "$probe_dir" 2>/dev/null || true
systemctl reset-failed kuno-training.service
systemctl restart kuno-training.service

echo "BlobFuse2 and training services installed; storage probe passed."