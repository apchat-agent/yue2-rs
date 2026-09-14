#!/usr/bin/env python3
"""Optional P3 first-velocity operation diagnostic; does not replace P0 fixtures."""
import os
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from dump_reference import snapshot
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.nar import CachedNAR, Chunk


@torch.inference_mode()
def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    root = Path.home() / "work/yue2-rs-fixtures/first-song"
    torch.set_num_threads(12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.set_per_process_memory_fraction(28 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    fixture = load_file(root / "nar.safetensors")
    stem = "two_chunk.0"
    model = YuE2ForCausalLM.from_pretrained(snapshot("YuE2-3B"), local_files_only=True,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    engine = CachedNAR(model, Chunk(fixture[stem+".ar_tokens"].tolist(), fixture[stem+".noise"]))
    result = {}
    def put(name, value):
        result[name] = value.detach().cpu().contiguous().clone()
    for i in [0,13,27]:
        put(f"cache.{i}.k", engine.cache[i][0]); put(f"cache.{i}.v", engine.cache[i][1])
    put("cos",engine.cos); put("sin",engine.sin); put("pos",engine.pos_emb)
    state = fixture[stem+".step.00.state"].cuda()
    raw = fixture[stem+".step.00.raw_t"].item()
    shifted = model._shift_t_value(raw, engine.device, engine.dtype)
    put("shifted", shifted)
    for index in [0,2]:
        model.time_embedder.mlp[index].register_forward_hook(lambda m,a,o,index=index: (put(f"time.{index}.input",a[0]), put(f"time.{index}.output",o)) and None)
    x = model.vae2llm(torch.nn.functional.pad(state, (0,0,1,1))[None]); put("vae2llm",x)
    time = model.time_embedder(shifted.expand(engine.nar_length))[None]; put("time",time)
    x = x + time; put("with_time",x)
    x = x + engine.pos_emb; put("injected",x)
    for i,(layer,(ar_k,ar_v)) in enumerate(zip(model.model.layers,engine.cache)):
        norm = layer.nar_input_layernorm(x)
        q,k,v = layer.nar_self_attn.project_qkv(norm,engine.cos,engine.sin)
        h = engine._attention(q[0],torch.cat((ar_k,k[0])),torch.cat((ar_v,v[0])))
        o = layer.nar_self_attn.o_proj(h.flatten(1)[None])
        if i == 0:
            for name,value in [("norm",norm),("q",q),("k",k),("v",v),("attn",h),("o",o)]: put("layer.0."+name,value)
        x = x + o
        x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
        put(f"layer.{i}.output",x)
    norm = model.model.norm(x); put("final_norm",norm)
    velocity = model.llm2vae(norm)[0,1:-1]; put("velocity",velocity)
    assert torch.equal(velocity.cpu(),fixture[stem+".step.00.first"])
    save_file(result,root / "nar-debug.safetensors")
    print(f"nar-debug.safetensors: {(root/'nar-debug.safetensors').stat().st_size:,} bytes; velocity exact against P0")


if __name__ == "__main__": main()
