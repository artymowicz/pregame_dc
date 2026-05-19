# Deploying the live bot

Target: a small always-on Linux VM (e.g. GCE `e2-small`, Debian 12). The bot
is a long-running daemon; it needs only the package and the shipped model
(`pregame_dc/models/dc_t-10min.npz`) — not the data parquets.

## 1. Get the code onto the host

The repo is private, so use a read-only GitHub **deploy key** generated on the
host (the private key never leaves the VM):

```bash
sudo apt-get update && sudo apt-get install -y git python3.11-venv
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/pregame_dc_deploy -N "" -C "gce deploy key"
cat ~/.ssh/pregame_dc_deploy.pub      # add at repo Settings > Deploy keys, read-only
printf 'Host github.com\n  IdentityFile ~/.ssh/pregame_dc_deploy\n  IdentitiesOnly yes\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config
ssh-keyscan github.com >> ~/.ssh/known_hosts
git clone git@github.com:artymowicz/pregame_dc.git
```

## 2. Install under /opt with a dedicated user

The venv has absolute paths baked in, so build it **after** moving the repo to
its final location:

```bash
sudo mv pregame_dc /opt/pregame_dc
sudo python3 -m venv /opt/pregame_dc/.venv
sudo /opt/pregame_dc/.venv/bin/pip install -e /opt/pregame_dc
sudo useradd --system --no-create-home --shell /usr/sbin/nologin pregamedc
sudo install -d -o pregamedc -g pregamedc /opt/pregame_dc/logs
sudo chown -R pregamedc:pregamedc /opt/pregame_dc
```

## 3. Credentials

Create `/opt/pregame_dc/.env` (never committed) with the wallet credentials:

```
POLYMARKET_PRIVATE_KEY=0x...
POLYMARKET_FUNDER_ADDRESS=0x...
```

then lock it down — it holds a real private key:

```bash
sudo chown pregamedc:pregamedc /opt/pregame_dc/.env
sudo chmod 600 /opt/pregame_dc/.env
```

## 4. systemd service

```bash
sudo cp /opt/pregame_dc/deploy/pregame-dc-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pregame-dc-bot
systemctl status pregame-dc-bot          # running?
journalctl -u pregame-dc-bot -f          # follow live output
```

Edit `ExecStart` in the unit file first if you want a different rule /
threshold / budget than the default (`--live --rule edge --threshold 0.04`).

## Updating

```bash
cd /opt/pregame_dc && sudo git pull
sudo /opt/pregame_dc/.venv/bin/pip install -e /opt/pregame_dc
sudo chown -R pregamedc:pregamedc /opt/pregame_dc
sudo systemctl restart pregame-dc-bot
```
