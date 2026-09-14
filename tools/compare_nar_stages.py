#!/usr/bin/env python3
"""Compare the P3b native-dtype dumps; diagnostics never relax the P3 gate."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from dump_nar_stages import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.home()/"work/yue2-rs-fixtures/first-song")
    args = parser.parse_args()
    torch.set_num_threads(12)
    python = load_file(args.root/"p3b-python.safetensors")
    rust = load_file(args.root/"p3b-rust.safetensors")
    metadata = json.loads((args.root/"p3b-python.json").read_text())
    rows = {}
    for name,actual in rust.items():
        reference_name = name.removeprefix("isolated.").removeprefix("own_rope.").removeprefix("reference_rope.")
        reference = python[reference_name]
        assert actual.shape == reference.shape,(name,actual.shape,reference.shape)
        assert actual.dtype == reference.dtype or (actual.dtype in (torch.uint32,torch.int64) and reference.dtype==torch.int64),(name,actual.dtype,reference.dtype)
        row = metrics(actual,reference)
        row.update(python_dtype=str(reference.dtype),rust_dtype=str(actual.dtype),shape=list(actual.shape))
        rows[name] = row
    def table(names):
        print("| Stage | Python/Rust dtype | max abs diff | cosine | differing / total |")
        print("| --- | --- | ---: | ---: | ---: |")
        for name in names:
            row = rows[name]
            dtype = row['python_dtype'].removeprefix('torch.')
            if row['rust_dtype'] != row['python_dtype']: dtype += '/' + row['rust_dtype'].removeprefix('torch.')
            print(f"| {name} | {dtype} | {row['max_abs']:.12g} | {row['cosine']:.12f} | {row['different']} / {row['count']} |")
    # Production execution order, including the invariant AR prefill dependency.
    stages = ["noise","state","ar.positions","nar.positions","audio.positions","audio.output","rope.inv_freq","ar.cos","ar.sin","ar.embedding.output"]
    stages += [f"ar.layer.00.{name}" for name in ("norm.output","q_proj.output","k_proj.output","v_proj.output","q_norm.output","k_norm.output","q","k","v","o_proj.input","o_proj.output")]
    stages += [f"ar.layer.{i:02}.output" for i in range(28)]
    stages += ["nar.cos","nar.sin","shifted","vae2llm.output","time.freqs","time.0.input","time.0.output","time.1.output","time.2.output","time.output","with_time","injected"]
    stages += [f"nar.layer.{i:02}.output" for i in range(28)]
    stages += ["final_norm.output","projection.output","velocity","latents"]
    table(stages)
    first = next(name for name in stages if rows[name]['max_abs'] > 1e-3)
    print(f"FIRST > 1e-3: {first}: {rows[first]['max_abs']:.12g}")
    print("\nSame-input operation controls:")
    table(sorted(name for name in rows if name.startswith('isolated.') and not name.startswith('isolated.step.')))
    print("\nSolver trajectory (actual Rust state, no resets):")
    print("| step | state | first velocity | mid | second velocity | next | next cosine |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for i in range(32):
        stem = f"step.{i:02}."
        values = ' | '.join(f"{rows[stem+s]['max_abs']:.12g}" for s in ('state','first','mid','second','next'))
        print(f"| {i} | {values} | {rows[stem+'next']['cosine']:.12f} |")
    exact_checks = [name for name in rows if name.startswith('isolated.step.') or name.startswith('shift.') or name in ('noise','state','audio.output') or name.endswith(('.positions','.shifted','.shifted_mid'))]
    assert all(rows[name]['max_abs']==0 for name in exact_checks),[(name,rows[name]['max_abs']) for name in exact_checks if rows[name]['max_abs']]
    raw_times = [name for name in rows if name.endswith(('.raw_t','.raw_mid'))]
    assert max(rows[name]['max_abs'] for name in raw_times) <= 1e-14
    print(f"EXACT checks: {len(exact_checks)}; raw-time max {max(rows[name]['max_abs'] for name in raw_times):.12g}")
    for name,row in metadata['controls'].items(): print(f"Python control {name}: {row}")
    for branch in ('ar','nar'):
        name = branch+'.layer.00.'
        print(f"Rust vs Python plain attention {branch}: {metrics(rust['isolated.'+name+'o_proj.input'],python['control.'+name+'plain_attention'])}")
    fp32 = {name:metrics(value,python[name]) for name,value in load_file(args.root/'p3b-rust-fp32.safetensors').items()}
    for name,row in fp32.items(): print(f"Rust vs Python {name}: {row}")
    (args.root/'p3b-comparison.json').write_text(json.dumps({'rows':rows,'first_over_1e-3':first,'python_controls':metadata['controls'],'fp32':fp32},indent=2)+'\n')
    size = sum(p.stat().st_size for p in args.root.rglob('*') if p.is_file())
    assert size <= 1_500_000_000,size
    print(f"Total recursive fixture bytes: {size:,}")


if __name__ == '__main__': main()
