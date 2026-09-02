import subprocess, os
MODEL = r"d:\Documents\PySrc\hayai\models\SmolLM2-135M-Instruct-Q4_K_M.gguf"
W = r"d:\Documents\pySrc\.scratch\w_smol2"
EMB = r"d:\Documents\PySrc\saor\target\release\embed_sparse.exe"
KLE = r"d:\Documents\PySrc\hayai\target\release\examples\kl_eval.exe"
P = r"d:\Documents\pySrc\.scratch\calib128.txt"
for sp in [0.10, 0.25]:
    spf = r"d:\Documents\pySrc\.scratch\smol_sp_f.txt"
    with open(spf, "w") as f:
        f.write("\n".join([f"{sp:.4f}"] * 30) + "\n")
    emb = r"d:\Documents\pySrc\.scratch\smol_sp_emb.gguf"
    if os.path.exists(emb):
        os.remove(emb)
    subprocess.run([EMB, "--model", MODEL, "--out", emb, "--weights", W, "--sparsities", spf], capture_output=True, text=True)
    r = subprocess.run([KLE, "--orig", MODEL, "--sparse", emb, "--prompts", P, "--n-positions", "4", "--device", "auto"], capture_output=True, text=True, timeout=1200)
    o = [l for l in (r.stdout + r.stderr).splitlines() if "kl_global" in l]
    print(f"sp={sp} -> {o[-1] if o else 'ERR ' + r.stderr[-120:]}")
    os.remove(emb)
