import argparse
import csv
import gc
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import textwrap

ROOT = Path('/mnt/data/home/sqkang/projects_yj/oppo\u5b66\u4e60/pixelDiT_explora')
REPO = ROOT / 'PixelDiT'
OUT = REPO / 'profile_results/pixeldit'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 't2i'))
PROMPT = 'A fluffy white cat sitting by a sunny window, detailed photography, warm natural light.'

def prepare_text():
    import torch
    from omegaconf import OmegaConf
    from diffusion.model.builder import get_tokenizer_and_text_encoder
    cfg = OmegaConf.load(REPO / 't2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml')
    tokenizer, encoder = get_tokenizer_and_text_encoder(device='cuda')
    prefix = '\n'.join(cfg.text_encoder.chi_prompt)
    length = len(tokenizer.encode(prefix)) + 300 - 2
    tokens = tokenizer([prefix + PROMPT], max_length=length, padding='max_length', truncation=True, return_tensors='pt').to('cuda')
    with torch.inference_mode():
        emb = encoder(tokens.input_ids, tokens.attention_mask)[0][:, [0] + list(range(-299, 0))]
    torch.save({'embedding': emb.cpu(), 'prompt': PROMPT}, OUT / 'text.pt')

def worker(name, width, height):
    import torch
    from omegaconf import OmegaConf
    import pixdit_core.pixeldit_t2i as module
    row = dict(name=name, width=width, height=height, patch=16, alignment='F16', image_tokens=width*height//256,
               text_tokens=300, batch=1, precision='bf16', cfg=False, warmup=1, repeats=3,
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               unit_memory='GiB', status='error', stage='load')
    path = OUT / (name + '.json')
    try:
        cfg = OmegaConf.load(REPO / 't2i/configs/PixelDiT_1024px_pixel_diffusion_stage3.yaml')
        net = module.PixDiT_T2I(**OmegaConf.to_container(cfg.model.extra)).eval().to(torch.bfloat16)
        ckpt = torch.load(ROOT / 'weights/pixeldit_t2i_v1.pth', map_location='cpu', weights_only=False)
        state = {k[5:]: v for k,v in ckpt['state_dict'].items() if k.startswith('core.')}
        net.load_state_dict(state, strict=True)
        del state, ckpt
        gc.collect()
        net.cuda()
        y = torch.load(OUT / 'text.pt', weights_only=False)['embedding'].cuda().to(torch.bfloat16)
        torch.manual_seed(2025)
        x = torch.randn(1,3,height,width,device='cuda',dtype=torch.bfloat16)
        t = torch.tensor([0.5],device='cuda',dtype=torch.bfloat16)
        original = net.forward
        source = textwrap.dedent(inspect.getsource(module.PixDiT_T2I.forward))
        source = source.replace('B, _, H, W = x.shape', 'self._stage = "encode"\n    self._events[0].record()\n    B, _, H, W = x.shape')
        source = source.replace('attn_mask_joint = None', 'self._events[1].record()\n        self._stage = "dit"\n        attn_mask_joint = None', 1)
        source = source.replace('x_pixels = self.final_layer(x_pixels)', 'self._events[2].record()\n    self._stage = "decode"\n    x_pixels = self.final_layer(x_pixels)')
        source = source.replace('return x_img', 'self._events[3].record()\n    return x_img')
        scope = dict(module.__dict__)
        exec(source, scope)
        import types
        net.forward = types.MethodType(scope['forward'], net)
        net._events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        row['stage'] = 'warmup'
        with torch.inference_mode():
            # Instrumentation must preserve the original forward numerically.
            if name == '1k':
                reference = original(x,t,y)
                measured = net(x,t,y)
                torch.cuda.synchronize()
                row['matches_original'] = bool(torch.equal(reference, measured))
                if not row['matches_original']:
                    raise RuntimeError('Instrumented output differs from original')
                del reference, measured
            else:
                result = net(x,t,y)
                torch.cuda.synchronize()
                del result
            # Verify actual SDPA backend in a separate run, outside timing.
            if name == '1k':
                from torch.profiler import profile, ProfilerActivity
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    result = net(x,t,y)
                    torch.cuda.synchronize()
                row['attention_ops'] = [e.key for e in prof.key_averages() if 'scaled_dot_product' in e.key]
                del result, prof
            torch.cuda.empty_cache()
            row['baseline_alloc_gib'] = torch.cuda.memory_allocated()/2**30
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(3):
                row['stage'] = 'measured_forward'
                result = net(x,t,y)
                torch.cuda.synchronize()
                ev = net._events
                timings.append([ev[i].elapsed_time(ev[i+1]) for i in range(3)] + [ev[0].elapsed_time(ev[3])])
                del result
        row.update(status='ok', stage='complete', timings_ms=timings)
        for i,key in enumerate(['encode_ms','dit_ms','decode_ms','total_ms']):
            row[key] = statistics.median(r[i] for r in timings)
    except Exception as exc:
        row['status'] = 'oom' if isinstance(exc, torch.OutOfMemoryError) else 'error'
        row['error'] = str(exc)
        if 'net' in locals(): row['stage'] = getattr(net, '_stage', row['stage'])
    finally:
        row['peak_alloc_gib'] = torch.cuda.max_memory_allocated()/2**30
        row['peak_reserved_gib'] = torch.cuda.max_memory_reserved()/2**30
        path.write_text(json.dumps(row,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--worker',nargs=3)
    p.add_argument('--prepare',action='store_true')
    args=p.parse_args()
    if args.prepare:
        prepare_text()
        return
    if args.worker:
        worker(args.worker[0],int(args.worker[1]),int(args.worker[2]))
        return
    if not (OUT/'text.pt').exists():
        subprocess.run([sys.executable,__file__,'--prepare'],check=True)
    cases=[('1k',1024,1024),('4k_typical',3360,2208),('4k_max',3360,2592),('8k',7680,4992)]
    rows=[]
    for name,w,h in cases:
        print('START',name,w,h,flush=True)
        with (OUT/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable,__file__,'--worker',name,str(w),str(h)],stdout=log,stderr=subprocess.STDOUT)
        path=OUT/(name+'.json')
        rows.append(json.loads(path.read_text()) if path.exists() else dict(name=name,status='process_failed'))
        print('DONE',name,rows[-1]['status'],flush=True)
    (OUT/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    fields=['name','width','height','patch','alignment','image_tokens','text_tokens','encode_ms','dit_ms','decode_ms','total_ms','peak_alloc_gib','peak_reserved_gib','status','stage']
    with (OUT/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    print('COMPLETE',flush=True)

if __name__=='__main__': main()

