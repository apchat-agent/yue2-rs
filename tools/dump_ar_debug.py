#!/usr/bin/env python3
"""Optional first-layer operation oracle for diagnosing the P1c numerical gate."""
import os
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from dump_reference import snapshot
from yue2.modeling_yue2 import YuE2ForCausalLM


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--investigate-gemm", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    root = Path.home() / "work/yue2-rs-fixtures/first-song"
    model = YuE2ForCausalLM.from_pretrained(snapshot("YuE2-3B"), local_files_only=True,
                                          dtype=torch.bfloat16).eval().cuda()
    tensors = {}
    layer = model.model.layers[0]
    for name, module in layer.named_modules():
        if name and not name.startswith("nar_"):
            def capture(_m, args, output, name=name):
                tensors[name + ".input"] = args[0].float().cpu()
                tensors[name + ".output"] = output.float().cpu()
            module.register_forward_hook(capture)
    project = layer.self_attn.project_qkv
    def project_qkv(x, cos, sin):
        tensors["cos"], tensors["sin"] = cos.cpu(), sin.cpu()
        q, k, v = project(x, cos, sin)
        tensors.update({"q_rot": q.float().cpu(), "k_rot": k.float().cpu(), "v": v.float().cpu()})
        return q, k, v
    layer.self_attn.project_qkv = project_qkv
    inputs = load_file(root / "ar_logits.safetensors")["input_ids"].cuda()
    model(inputs, use_cache=False)
    print("allow_bf16_reduced_precision_reduction:", torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        x = tensors[f"self_attn.{name}.input"].cuda()
        linear = getattr(layer.self_attn, name)
        precise = torch.nn.functional.linear(x, linear.weight.float()).bfloat16().float().cpu()
        original = tensors[f"self_attn.{name}.output"]
        print(name, "FP32 vs reference: exact", (precise == original).sum().item(), "/", original.numel(),
              "max", (precise - original).abs().max().item())
        tensors[f"self_attn.{name}.fp32"] = precise
        if name == "k_proj" and args.investigate_gemm:
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
            precise_bf16 = linear(x.bfloat16()).float().cpu()
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            print("k_proj without reduced precision", (precise_bf16 == original).sum().item(), "max", (precise_bf16-original).abs().max().item())
            for size in (256, 512, 1024):
                parts = [torch.nn.functional.linear(x[..., a:a+size], linear.weight[:, a:a+size].float()).bfloat16()
                         for a in range(0, x.shape[-1], size)]
                for acc in (torch.float32, torch.bfloat16):
                    reduced = torch.stack(parts).sum(0, dtype=acc).bfloat16().float().cpu()
                    print("k_proj split", size, acc, "exact", (reduced == original).sum().item(), "max", (reduced-original).abs().max().item())
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
                linear(x.bfloat16())
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
    (root / "debug").mkdir(exist_ok=True)
    save_file({k: v.contiguous().clone() for k, v in tensors.items()}, root / "debug/ar_operations.safetensors")
    print(f"Saved {len(tensors)} first-layer operation tensors")


if __name__ == "__main__":
    main()
