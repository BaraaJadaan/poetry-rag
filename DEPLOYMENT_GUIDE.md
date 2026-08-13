# Oracle Cloud Deployment Guide

This guide deploys the Poetry RAG service to the Oracle server you already have.
It assumes:

- Oracle Linux 7.9 is already installed (logged in as `opc`; confirmed via `cat /etc/os-release`).
- The server has 1 GB RAM (about 650 MB usable, as seen in `free -h`).
- You can already SSH into it.

Confirmed on the actual instance: it is **x86_64** (AMD/Intel), not ARM — the
free-tier `VM.Standard.E2.1.Micro` shape. Standard x86_64 Docker images and
prebuilt wheels work as-is; the ARM-compatibility concern only applies if you
later provision an Ampere A1 instance. An 8 GB swap file is already active, so
no swap setup is needed.

The OS is **Oracle Linux 7.9** (RHEL 7-compatible), which carries two
consequences:

- **It is past vendor maintenance.** Oracle Linux 7 general support ended in
  mid-2024; it now receives only paid extended support. Treat this box as a
  lab environment: keep the OpenRouter key server-side only (it already is),
  and rotate it if the instance is ever compromised. For a portfolio demo this
  is acceptable; if you want a maintained OS, rebuild the instance from the
  Oracle Linux 8 or 9 image instead and Section 2 reverts to the modern `dnf`
  form.
- **The toolchain is frozen at the RHEL 7 era:** `yum`, not `dnf`, and Docker
  stopped publishing new engine releases for RHEL/CentOS 7, so this guide pins
  the last supported 20.10.x line. That is fine for this project — nothing
  here needs a newer Docker feature.

## Important limitation: start small

Do not start by embedding the complete corpus on this 1 GB machine.

The cloud table is deliberately a *curated subset*: `POET_WHITELIST` in `embed_corpus.py`
restricts indexing to 40 poets (classical canon + most-quoted moderns + the poets the
Phase 6 eval set quotes), and a completeness filter drops single-hemistich and `...`
verses. The resulting ~350-420k verses still build to ~1.5-2 GB as 1024-dim float16
vectors (Matryoshka truncation — see Section 10), which is too much to hold in memory
at once. The safe first deployment is a small smoke test:

1. Install Docker and start the app.
2. Embed a small sample (`EMBED_LIMIT=1000` — that is 1,000 *poems*, ~22k verses, a
   few minutes of API calls), through OpenRouter.
3. Confirm the API, database, Cloudflare tunnel, and server Docker setup work.
4. Later, run the curated corpus (~420k verses, 6-10 hours) as a bounded-memory
   background job before considering anything larger.

Swap can reduce sudden out-of-memory failures, but it is slower than RAM and does not
make a full indexing job safe by itself.

## 1. Confirm SSH details

From your Windows PowerShell, connect as the default Oracle Linux user:

```powershell
ssh opc@<ORACLE_PUBLIC_IP>
```

Once connected, confirm the operating system, architecture, and memory — this
first step decides which Docker install path and which images apply, so do not
skip it:

```bash
cat /etc/os-release
uname -m
free -h
```

What this instance actually shows (recorded August 2026):

- `cat /etc/fedora-release` fails — there is no Fedora here. Oracle Linux
  writes `/etc/os-release` instead. On this box it reports `ID="ol"`,
  `VERSION_ID="7.9"`, and — the trap — `ID_LIKE="fedora"`. The RHEL 7 base
  means `yum`, not `dnf`, and RHEL 7-era packages. Detect the OS rather than
  assuming it from the shell prompt.
- `uname -m` shows `x86_64` — the free-tier E2.1.Micro shape is AMD/Intel, not
  the ARM Ampere A1. Standard x86_64 images apply.
- `free -h` shows ~668 MB total with 8 GB swap already active — the memory
  budget is even tighter than 1 GB suggests, and the swap section below is
  already satisfied.
- `df -h` shows a 39 GB root volume with only 19 GB free — the disk budget
  that decides the embedding storage strategy in Section 10.

If you ever SSH into an Ampere A1 instance instead, `uname -m` will show
`aarch64`; in that case build ARM64-compatible images and expect a
compile-from-source llama.cpp build inside the container.

## 2. Install Docker on Oracle Linux 7

Oracle Linux 7 is RHEL 7-compatible, and Docker never published a `rhel/7`
repo — the historical install path for the whole RHEL/CentOS 7 family is the
`centos/7` repo below. Docker stopped shipping new engine versions for el7,
so `yum install docker-ce` resolves to the last supported 20.10.x line. That
is acceptable here: the compose file uses only basic features, and Compose v2
is added separately as a standalone binary.

```bash
sudo yum -y install yum-utils device-mapper-persistent-data lvm2
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum -y install docker-ce docker-ce-cli containerd.io
sudo systemctl enable --now docker
sudo docker run hello-world
```

The `docker-compose-plugin` rpm does not exist for el7, so install Compose v2
as a CLI plugin binary instead. This keeps the `docker compose` (space) syntax
used everywhere in this guide:

```bash
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL -o /usr/local/lib/docker/cli-plugins/docker-compose https://github.com/docker/compose/releases/download/v2.24.7/docker-compose-linux-x86_64
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
```

If `yum install docker-ce` fails with a missing `container-selinux`
dependency, enable the Oracle repos that provide it and retry:

```bash
sudo yum-config-manager --enable ol7_optional_latest ol7_addons
sudo yum -y install container-selinux
```

Allow your normal SSH user to run Docker without `sudo`:

```bash
sudo usermod -aG docker "$USER"
exit
```

SSH into the server again, then verify:

```bash
docker run hello-world
docker compose version
```

## 3. Get the project onto the server

The preferred route is to clone the same GitHub repository that CI validates:

```bash
sudo yum -y install git
git clone <YOUR_REPOSITORY_URL> ~/poetry-rag
cd ~/poetry-rag
```

If the repository is private, use its SSH clone URL or another authenticated method.
Do not copy `.venv`, `models`, or a local `lancedb` directory from Windows.

## 4. Create the server environment file

Create `.env` on the server; never commit this file to GitHub:

```bash
cd ~/poetry-rag
vi .env
```

Use this content for the first smoke deployment:

```env
opentouter_api=YOUR_OPENROUTER_API_KEY
EMBED_LIMIT=1000
FRONTEND_ORIGINS=https://baraajadaan.github.io
CLOUDFLARED_TUNNEL_TOKEN=YOUR_CLOUDFLARE_TUNNEL_TOKEN
```

`CLOUD_DEPLOYMENT=true` is already set by `docker-compose.yml`, so the container uses
OpenRouter for generation and embeddings instead of downloading the local model.

The variable name `opentouter_api` contains a legacy spelling used by the current code.
Keep that exact spelling for now or the API embedding step will not find the key.

`FRONTEND_ORIGINS` is the origin allowed to call the backend from a browser. It contains
the GitHub Pages domain only; do not include the `/poetry-rag` path.

## 5. Build and run the smoke deployment

```bash
docker compose build
docker compose up -d
docker compose ps
```

The build is light on this box: the server dependencies no longer include
`llama-cpp-python` (removed in Phase 8.5), so `uv sync` inside the container
only downloads prebuilt x86_64 wheels — nothing compiles. If the daemon still
runs out of memory during the build, build the image on the Windows machine
instead and transfer it: `docker build --platform linux/amd64 --tag poetry-rag .`
on Windows, then `docker save poetry-rag | gzip > poetry-rag.tar.gz`, upload it,
and on the server `gunzip -c poetry-rag.tar.gz | docker load`, followed by
`docker compose up -d` (skip the build step).

Watch the web service while it creates the small LanceDB table:

```bash
docker compose logs -f web
```

The first run calls the OpenRouter embedding API for up to 1,000 poems (the `limit`
counts *matching* poems after the whitelist filter — about 22k verses, roughly 15-25
minutes of API calls). When it finishes, the FastAPI service starts.

The database and models are stored in Docker named volumes, so they survive a normal
container restart.

## 6. Create a stable Cloudflare backend URL

The GitHub Pages frontend needs a backend URL that does not change after a restart. For
the named tunnel route below, you need a domain/zone managed by Cloudflare. If you do not
have one, you can use the temporary Quick Tunnel described later, but the GitHub Pages
site will stop working whenever its random URL changes.

In the [Cloudflare dashboard](https://dash.cloudflare.com/):

1. Go to **Networking → Tunnels** and select **Create a tunnel**.
2. Name it `poetry-rag-api` and create it.
3. Choose **Docker** as the connector.
4. Copy only the tunnel token from the Docker command. Do not commit it to GitHub.
5. Add a **Published application** route for the tunnel.
6. Choose a hostname such as `poetry-api.example.com`.
7. Set the service URL to `http://web:8000`.
8. Save the route.

The service URL must be `http://web:8000`, not `http://localhost:8000`: Cloudflared is a
separate Compose container and reaches FastAPI through Docker's service name `web`.

Put the copied token in the server's `.env`:

```env
CLOUDFLARED_TUNNEL_TOKEN=YOUR_CLOUDFLARE_TUNNEL_TOKEN
```

Then start the named tunnel:

```bash
docker compose up -d --force-recreate cloudflared
docker compose logs --tail=100 cloudflared
```

The tunnel should show a healthy/connected status in Cloudflare. Test the public backend:

```bash
curl https://poetry-api.example.com/config
```

You should receive JSON containing `cloud_deployment: true`.

Cloudflare's official dashboard flow is documented here:
[Create a remotely-managed tunnel](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/)

## 7. Open the app through Cloudflare

Read the tunnel logs if you need to diagnose a connection:

```bash
docker compose logs cloudflared
```

Open the stable hostname you configured, such as `https://poetry-api.example.com`.
The FastAPI port is bound only to `127.0.0.1` on the VM; Cloudflared is the public entry
point, so you do not need to open port 8000 in Oracle networking for this setup.

Useful checks:

```bash
curl http://127.0.0.1:8000/health
docker stats
df -h
free -h
```

## 8. Stop, restart, and inspect the service

```bash
docker compose restart
docker compose logs --tail=100 web
docker compose logs --tail=100 cloudflared
docker compose down
```

`docker compose down` keeps the named volumes. Be careful with:

```bash
docker compose down -v
```

The `-v` option deletes the persistent LanceDB volume and therefore deletes the sample
index. Use it only when you intentionally want to rebuild the database.

## 9. Deploy the frontend to GitHub Pages

GitHub Pages hosts the static files in `frontend/`; it does not run FastAPI or Docker.
The included `.github/workflows/pages.yml` publishes the frontend and injects the backend
URL into `frontend/config.js` during deployment.

Use the stable HTTPS URL created in Section 6.

Then, on GitHub:

1. Open the repository's **Settings → Secrets and variables → Actions → Variables**.
2. Create a repository variable named `BACKEND_URL`.
3. Set its value to the stable backend URL, such as `https://poetry-api.example.com`.
4. Open **Settings → Pages** and choose **GitHub Actions** as the source.
5. Push a change to `main`, or run **Deploy frontend to GitHub Pages** manually from the
   Actions tab.

The project site URL will normally be:

```text
https://baraajadaan.github.io/poetry-rag/
```

The browser will load the interface from GitHub Pages and send `GET /config` and
`POST /chat` to the Oracle backend. The backend keeps the OpenRouter key private.

If you change the Pages domain or use a custom domain, update `FRONTEND_ORIGINS` in the
server's `.env` and restart the web container:

```bash
docker compose up -d --force-recreate web
```

## 10. What comes next for the full (curated) corpus

`EMBED_LIMIT` is set on the server. The cloud table is a curated subset by design —
`POET_WHITELIST` (40 poets) plus a completeness filter, both wired only when
`USE_OPENROUTER_EMBED=true` — so removing the limit starts the *filtered* pipeline, not
the raw 3.4M-verse corpus. The filtered run is safe to restart only because the
pipeline streams source rows instead of loading them, but it is still hours of API
calls, so keep `EMBED_LIMIT=1000` during setup.

The curated-corpus plan was checked against the server's real limits (the full source
is 3,449,196 verses; the curated subset is ~350-420k after whitelist + quality filter):

- **Disk is comfortable now, thanks to the curation.** `df -h` on the actual box:
  39 GB root volume with only 19 GB free (52% used already; the container image takes
  another ~2 GB). At the model's default 4096 dimensions even the full corpus would be
  80-100 GB — impossible — and the truncation+float16 pair (Matryoshka: Qwen3-Embedding
  keeps retrieval quality in its first 1024 dims; f16 storage verified against LanceDB:
  f16 columns search correctly with an f32 query) brings the full corpus to ~10 GB, so
  the curated subset lands at ~1.5-2 GB. Cost is a non-issue: at $0.01/M input tokens
  the whole corpus is under $1 — the curated run costs mere cents. The smoke-test
  table is created with this exact final schema (`vec[:1024]` + f16), so nothing needs
  re-embedding later. If more headroom is ever wanted, OCI allows resizing the boot
  volume online (the free-tier block-storage quota is 200 GB total) without
  reinstalling.
- **RAM: stay bounded.** The pipeline streams source rows and keeps only the
  deduplicated chunk set + existing-ID set in memory (~2-3 GB peak for the curated
  run — borderline on a 1 GB box, which is why the smoke test comes first). For the
  real run, either raise the box's swap already in place (8 GB active) or split the
  curation into a manifest-based script: filter the corpus once to a cached file, then
  embed batch-by-batch (batches of 100, existing SHA-256 IDs skipped), appending to
  LanceDB and dropping each window. Record progress in a resumable manifest so a
  failed run restarts at the first missing ID rather than re-querying.
- **Throughput: expect a long background job.** At ~17 verses/sec the curated 420k
  verses take 6-10 hours of continuous calls; the full corpus would take 2-3 days.
  Do not watch interactively — run it detached and check progress daily. If too slow,
  raise concurrency with a small worker pool (the resumable manifest makes parallel
  workers safe) — the API is the bottleneck, not the box.
- **Query speed: optional ANN index.** A brute-force scan of ~1.5-2 GB (350k × 1024
  dims f16) per query is measurable but acceptable on this hardware; if it is not,
  `create_index` with an IVF/PQ config brings queries to tens of milliseconds. The
  smoke-test table does not need one and must not get full-corpus treatment by
  accident.

Sequence: (1) embed the 1,000-poem smoke table and validate end-to-end; (2) benchmark
a larger limit (e.g. 50,000 poems) against `free -h`/`df -h`; (3) kick the curated run
in `tmux`/`screen` and check progress daily; (4) re-run the Phase 6 evaluation against
the cloud table — all 18 golden eval verses are in the curated subset, so the harness
works unmodified.

## Troubleshooting

If the web container exits:

```bash
docker compose logs web
```

Common causes are a missing `opentouter_api` key, an OpenRouter quota/API error, or an
out-of-memory kill during indexing.

If Docker itself is not running:

```bash
sudo systemctl status docker
sudo systemctl restart docker
```

If Oracle Linux 7 reports an iptables/firewalld backend problem after Docker
installation, check Docker's CentOS/RHEL 7 troubleshooting material before
changing firewall rules. Do not expose port 8000 publicly when the Cloudflare
tunnel is being used.

If you need a temporary URL before setting up a domain, restore the Quick Tunnel command
temporarily in `docker-compose.yml`:

```yaml
command: tunnel --url http://web:8000 --no-autoupdate
```

Then run `docker compose up -d --force-recreate cloudflared` and read its random
`trycloudflare.com` URL. This is for testing only; replace it with the named tunnel before
configuring the GitHub Pages `BACKEND_URL` variable.
