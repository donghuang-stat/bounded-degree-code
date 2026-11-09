#!/usr/bin/env python3
"""Helper for the SLURM scripts.  Paths in the config (out_dir, cache_dir) that are
relative are interpreted relative to the package directory, because every script
changes into that directory before running anything.

    cfginfo.py info CONFIG.json               shell assignments (eval-able)
    cfginfo.py missing CONFIG.json [--chunk L]
                                              missing shard indices, one line per array chunk:
                                              "OFFSET SPEC COUNT" with SPEC a SLURM --array
                                              range list of local indices (shard = OFFSET + index,
                                              0 <= index < L)
    cfginfo.py prebuild CONFIG.json           build (or load) the motif library cache
    cfginfo.py chunks N L                     all shards 0..N-1 in the "missing" format
    cfginfo.py campaign CAMPAIGN.json         shell assignments CAMP_NAME, CAMP_OUT, CONFIGS
                                              (bash array of absolute config paths)
Only 'prebuild' needs numpy/pynauty.
"""
import json, math, os, shlex, sys, time

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# number of connected graphs with m edges and no isolated vertices (OEIS A002905)
N_CONNECTED = {1: 1, 2: 1, 3: 3, 4: 5, 5: 12, 6: 30, 7: 79, 8: 227, 9: 710}


def load(path):
    with open(path) as f:
        cfg = json.load(f)
    return cfg


def resolve(p):
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(PKG, p))


def paths(cfg):
    out_dir = resolve(cfg["out_dir"])
    cache_dir = resolve(cfg.get("cache_dir", os.path.join(PKG, "cache")))
    cache = os.path.join(cache_dir, f"motiflib_{cfg['mmin']}_{cfg['mmax']}.pkl")
    return out_dir, cache_dir, cache


def nshards(cfg):
    T = sum(cfg["reps"])
    return (T + cfg["shard_size"] - 1) // cfg["shard_size"]


def ranges(idx):
    """[0,1,2,5,7,8] -> '0-2,5,7-8'"""
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


def cmd_info(cfg, cfgpath):
    out_dir, cache_dir, cache = paths(cfg)
    npairs = sum(cfg["reps"])
    nmot = sum(N_CONNECTED[m] for m in range(cfg["mmin"], cfg["mmax"] + 1))
    # aggregate.py streams the shards and keeps n_motifs products per pair; the family
    # statistics and rank computations add a modest overhead
    raw = npairs * 2 * nmot * 8
    agg_gb = int(math.ceil(1.5 * (raw / 2) / 2**30 + 4))
    tag = "mot_" + os.path.splitext(os.path.basename(cfgpath))[0]
    for k, v in dict(NSHARDS=nshards(cfg), NPAIRS=npairs, NMOTIFS=nmot, OUT_DIR=out_dir,
                     CACHE_DIR=cache_dir, CACHE_FILE=cache, CACHE_EXISTS=int(os.path.exists(cache)),
                     JOB_TAG=tag, AGG_MEM_AUTO=f"{agg_gb}G",
                     RAW_GB=f"{raw / 1e9:.2f}").items():
        print(f"{k}={shlex.quote(str(v))}")


def print_chunks(miss, chunk):
    groups = {}
    for k in miss:
        groups.setdefault(k // chunk, []).append(k % chunk)
    for g in sorted(groups):
        print(g * chunk, ranges(groups[g]), len(groups[g]))


def cmd_missing(cfg, chunk):
    out_dir, _, _ = paths(cfg)
    N = nshards(cfg)
    miss = [k for k in range(N) if not os.path.exists(os.path.join(out_dir, f"shard_{k:05d}.npz"))]
    print_chunks(miss, chunk)


def cmd_prebuild(cfg):
    sys.path.insert(0, PKG)
    from motiflib import MotifLibrary
    _, cache_dir, cache = paths(cfg)
    t0 = time.time()
    existed = os.path.exists(cache)
    lib = MotifLibrary(cfg["mmin"], cfg["mmax"], cache_dir=cache_dir, verbose=True)
    print(f"[prebuild] {'loaded' if existed else 'built'} {cache}: {len(lib.motifs)} motifs, "
          f"{len(lib.nodes)} DAG nodes, {time.time() - t0:.1f}s", flush=True)


def campaign_configs(path):
    """Same resolution as campaign_plot.load_campaign: a config path is relative to the
    package directory if that file exists, otherwise relative to the campaign file."""
    camp = load(path)
    if "configs" not in camp:
        return camp.get("name", "single"), resolve(camp["out_dir"]), [os.path.abspath(path)]
    base = os.path.dirname(os.path.abspath(path))
    out = []
    for c in camp["configs"]:
        q = c if os.path.isabs(c) else (resolve(c) if os.path.exists(resolve(c)) else os.path.join(base, c))
        if not os.path.exists(q):
            raise SystemExit(f"config {c} of {path} not found")
        out.append(q)
    return camp.get("name", os.path.splitext(os.path.basename(path))[0]), resolve(camp["out_dir"]), out


def cmd_campaign(path):
    name, out, cfgs = campaign_configs(path)
    print(f"CAMP_NAME={shlex.quote(name)}")
    print(f"CAMP_OUT={shlex.quote(out)}")
    print("CONFIGS=(" + " ".join(shlex.quote(c) for c in cfgs) + ")")


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "missing", "prebuild", "chunks", "campaign"):
        raise SystemExit(__doc__)
    cmd, cfgpath = sys.argv[1], sys.argv[2]
    if cmd == "chunks":
        # equal-sized chunks (each <= L), so that equally throttled arrays finish together
        N, L = int(sys.argv[2]), int(sys.argv[3])
        L = -(-N // -(-N // L)) if N > 0 else L
        print_chunks(list(range(N)), L)
        return
    if cmd == "campaign":
        cmd_campaign(cfgpath)
        return
    cfg = load(cfgpath)
    if cmd == "info":
        cmd_info(cfg, cfgpath)
    elif cmd == "missing":
        chunk = int(sys.argv[sys.argv.index("--chunk") + 1]) if "--chunk" in sys.argv else 10**9
        cmd_missing(cfg, chunk)
    else:
        cmd_prebuild(cfg)


if __name__ == "__main__":
    main()
