"""P3b diagnostics, invoked by dump_reference.py --nar-stages.

Hooks observe the installed implementation. Only the explicitly named numeric
controls alter arithmetic; the immutable P0 oracle is checked, never replaced.
"""
import math

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.nar import CachedNAR, Chunk, song_chunks


def metrics(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    assert a.shape == b.shape and a.isfinite().all() and b.isfinite().all()
    return {"max_abs": (a-b).abs().max().item(),
            "cosine": F.cosine_similarity(a, b, dim=0).item(),
            "different": (a != b).sum().item(), "count": a.numel()}


def plain_attention(q, k, v, causal=False):
    """Diagnostic transcription of candle's unfused attention arithmetic."""
    query = q.transpose(0,1)[None]
    groups = q.shape[1] // k.shape[1]
    key = k.transpose(0,1)[None].repeat_interleave(groups,1).float().transpose(-1,-2).contiguous()
    value = v.transpose(0,1)[None].repeat_interleave(groups,1).float().contiguous()
    outputs = []
    for start in range(0,len(q),128):
        count = min(128,len(q)-start)
        scores = query[...,start:start+count,:].float().contiguous() @ key / (q.shape[-1] ** 0.5)
        if causal:
            visible = torch.arange(len(k),device=q.device)[None,:] <= torch.arange(start,start+count,device=q.device)[:,None]
            scores = scores.masked_fill(~visible,float('-inf'))
        numerator = (scores-scores.amax(-1,keepdim=True)).exp()
        result = (numerator.to(q.dtype).float() @ value) / numerator.sum(-1,keepdim=True)
        outputs.append(result.to(q.dtype))
    return torch.cat(outputs,-2)[0].transpose(0,1)


@torch.inference_mode()
def dump_nar_stages(writer, model_dir):
    from dump_reference import trace_chunk, sha256
    fixture_path = writer.directory / "nar.safetensors"
    original_hash = sha256(fixture_path)
    fixture = load_file(fixture_path)
    stem = "two_chunk.0."
    chunk = Chunk(fixture[stem+"ar_tokens"].tolist(), fixture[stem+"noise"])
    model = YuE2ForCausalLM.from_pretrained(model_dir, local_files_only=True,
             dtype=torch.bfloat16, low_cpu_mem_usage=True).eval().cuda()
    tensors, order, hooks, originals = {}, [], [], []
    def put(name, value):
        if name not in tensors:
            order.append(name)
        tensors[name] = value.detach().cpu().contiguous().clone()
    def hook(module, name, inputs=False):
        def capture(_m, a, out):
            if inputs: put(name+".input", a[0])
            put(name+".output", out)
        hooks.append(module.register_forward_hook(capture))
    put("noise", chunk.noise)
    put("state", chunk.noise.cuda().to(model.dtype))
    put("ar.positions", torch.arange(len(chunk.ar_tokens), dtype=torch.int64)[None])
    put("nar.positions", torch.arange(len(chunk.ar_tokens), len(chunk.ar_tokens)+len(chunk.noise)+2)[None])
    put("audio.positions", torch.arange(len(chunk.noise)+2))
    hook(model.model.embed_tokens, "ar.embedding")
    hook(model.latent_pos_embed, "audio", inputs=True)
    for i, layer in enumerate(model.model.layers):
        for branch in ("ar", "nar"):
            prefix = f"{branch}.layer.{i:02}"
            norm = layer.post_attention_layernorm if branch == "ar" else layer.nar_pre_mlp_layernorm
            mlp = layer.mlp if branch == "ar" else layer.nar_mlp
            residual = {}
            def pre(_m, a, residual=residual): residual["x"] = a[0]
            def post(_m, _a, out, residual=residual, prefix=prefix):
                put(prefix+".output", residual.pop("x") + out)
            hooks.append(norm.register_forward_pre_hook(pre))
            hooks.append(mlp.register_forward_hook(post))
            if i != 0: continue
            input_norm = layer.input_layernorm if branch == "ar" else layer.nar_input_layernorm
            attention = layer.self_attn if branch == "ar" else layer.nar_self_attn
            hook(input_norm, prefix+".norm", inputs=True)
            for name in ("q_proj", "k_proj", "v_proj", "q_norm", "k_norm", "o_proj"):
                hook(getattr(attention, name), prefix+"."+name, inputs=True)
            project = attention.project_qkv
            def projected(x, cos, sin, project=project, prefix=prefix, branch=branch):
                put(branch+".cos", cos); put(branch+".sin", sin)
                q,k,v = project(x,cos,sin)
                for name,value in zip(("q","k","v"),(q,k,v)): put(prefix+"."+name,value)
                return q,k,v
            originals.append((attention, "project_qkv", project))
            attention.project_qkv = projected
    hook(model.vae2llm, "vae2llm", inputs=True)
    for i in (0,1,2): hook(model.time_embedder.mlp[i], f"time.{i}", inputs=True)
    hook(model.time_embedder, "time", inputs=True)
    hook(model.model.norm, "final_norm", inputs=True)
    hook(model.llm2vae, "projection", inputs=True)
    engine = CachedNAR(model, chunk)
    put("rope.inv_freq", model.model.rotary_emb._inv_freq)
    put("time.freqs",torch.exp(-math.log(10000)*torch.arange(128,dtype=torch.float32,device=model.device)/128))
    for i in (0,13,27):
        for name,value in zip(("k","v"),engine.cache[i]): put(f"cache.{i:02}.{name}",value[None])
    put("shifted", model._shift_t_value(20.,engine.device,engine.dtype))
    actual = engine.velocity(tensors["state"].cuda(),20.)
    put("velocity",actual)
    assert torch.equal(actual.cpu(),fixture[stem+"step.00.first"])
    for h in hooks: h.remove()
    for obj,name,original in originals: setattr(obj,name,original)
    # Explicit additive stages; check against the actual layer-0 input hook.
    put("with_time", tensors["vae2llm.output"] + tensors["time.output"][None])
    put("injected", tensors["with_time"] + tensors["audio.output"][None])
    assert torch.equal(tensors["injected"],tensors["nar.layer.00.norm.input"])
    engine.close()
    loop = trace_chunk(model,chunk,32)
    for name,value in loop.items():
        if name.startswith("step.") or name == "latents":
            assert torch.equal(value,fixture[stem+name]), name
            put(name,value)
    for step in range(32):
        for suffix,raw_suffix in (("shifted","raw_t"),("shifted_mid","raw_mid")):
            raw = loop[f"step.{step:02}.{raw_suffix}"].item()
            put(f"step.{step:02}.{suffix}",model._shift_t_value(raw,model.device,model.dtype))
    # Check the release's RNG ownership/order using a fresh full-song draw.
    token_fixture = load_file(writer.directory / "tokens.safetensors")
    chunks = song_chunks(token_fixture["prefix"].tolist(),token_fixture["semantic_ids"].tolist(),831001,2098)
    assert all(torch.equal(c.noise,fixture[f"two_chunk.{i}.noise"]) for i,c in enumerate(chunks))
    print("P3b original velocity, all 32 solver steps, two full-song noise slices: EXACT",flush=True)
    controls = {}
    for branch in ("ar","nar"):
        prefix = branch+".layer.00."
        q,k,v = (tensors[prefix+name][0].cuda() for name in ("q","k","v"))
        if branch == "nar":
            k = torch.cat((tensors["cache.00.k"][0].cuda(),k))
            v = torch.cat((tensors["cache.00.v"][0].cuda(),v))
        plain = plain_attention(q,k,v,causal=branch=="ar").flatten(1)[None].cpu()
        controls[prefix+"plain_attention"] = metrics(plain,tensors[prefix+"o_proj.input"])
        put("control."+prefix+"plain_attention",plain)
    # Isolate GEMM: same BF16 inputs/weights, only reduction precision changes.
    for name,module in (("vae2llm",model.vae2llm),("time.0",model.time_embedder.mlp[0]),
                        ("time.2",model.time_embedder.mlp[2]),
                        ("nar.layer.00.v_proj",model.model.layers[0].nar_self_attn.v_proj)):
        x = tensors[name+".input"].cuda()
        reference = tensors[name+".output"]
        fp32 = F.linear(x.float(),module.weight.float(), None if module.bias is None else module.bias.float()).bfloat16().cpu()
        controls[name+".fp32"] = metrics(fp32,reference)
        put("control."+name+".fp32",fp32)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        full_accum = module(x).cpu()
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        controls[name+".no_reduced"] = metrics(full_accum,reference)
        put("control."+name+".no_reduced",full_accum)
    # Same mathematical model, noise, steps and masks. Never use these as a new oracle.
    for variant in ("no_reduced", "math_sdpa", "plain_sdpa"):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = variant != "no_reduced"
        for i in range(2):
            c = Chunk(fixture[f"two_chunk.{i}.ar_tokens"].tolist(),fixture[f"two_chunk.{i}.noise"])
            e = CachedNAR(model,c,attention="math" if variant == "math_sdpa" else "sdpa")
            if variant == "plain_sdpa":
                # Rebuild the AR prefill using the same altered arithmetic too.
                e.cache.clear()
                e._attention = plain_attention
                e._prefill()
            result = e.solve(32)
            controls[f"{variant}.chunk.{i}"] = metrics(result,fixture[f"two_chunk.{i}.latents"])
            put(f"control.{variant}.chunk.{i}",result)
            print(f"P3b numeric control {variant} chunk {i}: {controls[f'{variant}.chunk.{i}']}",flush=True)
            e.close()
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    # Cover shift != 1 without changing the checkpoint's production configuration.
    raw = torch.tensor([20., 4., 1., 0., -1., -4., -20.],dtype=torch.float64)
    put("shift.raw",raw)
    for shift in (0.5,1.,3.):
        model.config.timestep_shift = shift
        values = torch.stack([model._shift_t_value(t.item(),model.device,model.dtype) for t in raw])
        put(f"shift.{shift:g}",values)
    model.config.timestep_shift = 1.
    # Cross-language FP32 control: same checkpoint values, masks and solver.
    # This is NOT gate P3, whose BF16 oracle stays immutable.
    parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    buffer_dtypes = {name:str(value.dtype) for name,value in model.named_buffers()}
    model.float()
    for i in range(2):
        c = Chunk(fixture[f"two_chunk.{i}.ar_tokens"].tolist(),fixture[f"two_chunk.{i}.noise"])
        e = CachedNAR(model,c,attention="math")
        result = e.solve(32)
        put(f"control.fp32.chunk.{i}",result)
        print(f"P3b FP32 control chunk {i} complete",flush=True)
        e.close()
    writer.tensors("p3b-python.safetensors",tensors)
    assert sha256(fixture_path) == original_hash
    writer.json("p3b-python.json", {"order":order,"controls":controls,"files":writer.files,
        "nar_sha256":original_hash,"checkpoint_config":model.config.to_dict(),
        "notes":"Original production CachedNAR observed with hooks; BF16 stages kept BF16; controls never replace P0",
        "norm_eps":{name:module.eps for name,module in model.named_modules() if hasattr(module,"eps")},
        "parameter_dtypes":parameter_dtypes,
        "buffer_dtypes":buffer_dtypes,
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
        "peak_cuda_reserved_bytes":torch.cuda.max_memory_reserved()})
    size = sum(p.stat().st_size for p in writer.directory.rglob("*") if p.is_file())
    print(f"Total recursive fixture bytes: {size:,}; peak CUDA allocated {torch.cuda.max_memory_allocated():,}",flush=True)
    assert size <= 1_500_000_000
