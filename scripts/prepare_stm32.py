"""Export and host-verify the existing speech graph for STM32F103C8T6.

No hardware is accessed. Firmware sizes are linker allocations, not measured
MCU peak SRAM. This is a separate folded-BN integer deployment backend.
"""
import argparse
import ctypes
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from cig_mpq.models import AudioDSCNN
from cig_mpq import mcu

FW=ROOT/'firmware/stm32f103'
NAMES=['features.0','features.4.depthwise','features.4.pointwise','features.6.depthwise',
       'features.6.pointwise','features.7.depthwise','features.7.pointwise']
RELUS=[True,False,True,False,True,False,True]
POOLS=[2,1,2,1,1,1,1]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weight_quantization(weight,bits):
    maximum=(1 << (bits-1))-1
    flat=np.asarray(weight,dtype=np.float64).reshape(weight.shape[0],-1)
    scale=np.maximum(np.max(np.abs(flat),axis=1),1e-12)/maximum
    values=np.clip(np.rint(flat/scale[:,None]),-(1 << (bits-1)),maximum).astype(np.int8)
    return values.reshape(weight.shape),scale


def fixed_constants(bias,input_scale,weight_scale,output_scale,fan_in,bits):
    if bias is None: bias=np.zeros_like(weight_scale)
    values=np.rint(np.asarray(bias,dtype=np.float64)/(input_scale*weight_scale))
    bound=np.abs(values)+fan_in*255*(1 << (bits-1))
    if not np.isfinite(values).all() or np.any(bound > np.iinfo(np.int32).max):
        raise ValueError('int32 accumulator bound exceeded')
    parameters=[mcu.multiplier(float(s*input_scale/output_scale)) for s in weight_scale]
    return values.astype(np.int32),np.array([p[0] for p in parameters],dtype=np.int32),np.array([p[1] for p in parameters],dtype=np.uint8)


def export_model(checkpoint,features,mask,device):
    model=AudioDSCNN().eval()
    model.load_state_dict(checkpoint['state_dict'])
    model.features[0]=fuse_conv_bn_eval(model.features[0],model.features[1])
    model.features[1]=nn.Identity()
    for index in (4,6,7):
        block=model.features[index]
        block.pointwise=fuse_conv_bn_eval(block.pointwise,block.norm)
        block.norm=nn.Identity()
    modules=dict(model.named_modules())
    quantized={}
    for index,name in enumerate(NAMES):
        bits=4 if mask & (1 << index) else 8
        module=modules[name]
        weight,scale=weight_quantization(module.weight.detach().numpy(),bits)
        quantized[name]=(weight,scale,bits)
        module.weight.data.copy_(torch.from_numpy(weight.astype(np.float32)*scale.reshape(-1,1,1,1)).float())
    fc_weight,fc_scale=weight_quantization(model.classifier.weight.detach().numpy(),8)
    model.classifier.weight.data.copy_(torch.from_numpy(fc_weight*fc_scale[:,None]).float())
    ranges={name:[0.,0.] for name in NAMES}
    handles=[]
    def observer(name,relu):
        def hook(_m,_a,out):
            if relu: out=out.clamp_min(0)
            ranges[name][0]=min(ranges[name][0],float(out.min()))
            ranges[name][1]=max(ranges[name][1],float(out.max()))
        return hook
    for name,relu in zip(NAMES,RELUS): handles.append(modules[name].register_forward_hook(observer(name,relu)))
    model.to(device)
    with torch.inference_mode():
        for start in range(0,len(features),128):
            model(torch.from_numpy(features[start:start+128]).unsqueeze(1).to(device))
    for handle in handles: handle.remove()
    model.cpu()
    input_scale,input_zero=mcu.affine_parameters(float(features.min()),float(features.max()))
    previous_scale,previous_zero=input_scale,input_zero
    layers=[]
    height,width=49,10
    for index,name in enumerate(NAMES):
        module=modules[name]
        weight,scale,bits=quantized[name]
        output_scale,output_zero=mcu.affine_parameters(*ranges[name])
        bias,mult,shift=fixed_constants(None if module.bias is None else module.bias.detach().numpy(),previous_scale,scale,
            output_scale,weight[0].size,bits)
        layers.append(dict(name=name,weights=weight.transpose(0,2,3,1).copy(),bits=bits,
            bias=bias,mult=mult,shift=shift,in_zero=previous_zero,out_zero=output_zero,
            depthwise=module.groups>1,relu=RELUS[index],pool=POOLS[index],
            height=height,width=width,in_c=module.in_channels,out_c=module.out_channels,
            input_scale=previous_scale,output_scale=output_scale,observed_range=ranges[name]))
        height,width=height//POOLS[index],width//POOLS[index]
        previous_scale,previous_zero=output_scale,output_zero
    bias,mult,shift=fixed_constants(model.classifier.bias.detach().numpy(),previous_scale,
                                  fc_scale,1/1024,128,8)
    fc=dict(weights=fc_weight,bias=bias,mult=mult,shift=shift,in_zero=previous_zero)
    return layers,fc,input_scale,input_zero


def emit_data(output,layers,fc,mask,input_scale,input_zero):
    chunks=['#include "mcu.h"\n']
    arrays={}
    metadata=[]
    def array(name,values,dtype):
        values=np.asarray(values,dtype=dtype).reshape(-1)
        arrays[name]=values
        typ={'uint8':'uint8_t','int32':'int32_t'}[str(values.dtype)]
        chunks.append(f'const {typ} {name}[{len(values)}]={{\n')
        for start in range(0,len(values),24): chunks.append(','.join(str(int(v)) for v in values[start:start+24])+',\n')
        chunks.append('};\n')
    for index,layer in enumerate(layers):
        raw=layer['weights'].reshape(-1)
        payload=mcu.pack_int4(raw) if layer['bits']==4 else raw.tobytes()
        array(f'l{index}_w',np.frombuffer(payload,dtype=np.uint8),'uint8')
        for key,dtype in (('bias','int32'),('mult','int32'),('shift','uint8')):
            array(f'l{index}_{key}',layer[key],dtype)
        metadata.append({k:v for k,v in layer.items() if k not in ('weights','bias','mult','shift')})
        metadata[-1]['weight_payload_bytes']=len(payload)
    array('mcu_fc_weights',fc['weights'].view(np.uint8),'uint8')
    for key,dtype in (('bias','int32'),('mult','int32'),('shift','uint8')):
        array(f'mcu_fc_{key}',fc[key],dtype)
    identity=hashlib.sha256(json.dumps(dict(mask=mask,input_scale=input_scale,input_zero=input_zero,
        layers=metadata,fc_zero=fc['in_zero'],logits_scale=1/1024),sort_keys=True).encode())
    for name,values in sorted(arrays.items()): identity.update(name.encode()+values.tobytes())
    model_hash=identity.hexdigest()
    array('mcu_model_hash',np.frombuffer(bytes.fromhex(model_hash),dtype=np.uint8),'uint8')
    chunks.append(f'const uint32_t mcu_mask={mask};\nconst int32_t mcu_fc_zero={fc["in_zero"]};\n')
    chunks.append('const MCULayer mcu_layers[7]={\n')
    for index,layer in enumerate(layers):
        fields=[layer[k] for k in ('height','width','in_c','out_c')]
        fields += [layer['weights'].shape[1],int(layer['depthwise']),layer['pool'],layer['bits'],int(layer['relu']),layer['in_zero'],layer['out_zero']]
        chunks.append('{'+','.join([f'l{index}_w',f'l{index}_bias',f'l{index}_mult',f'l{index}_shift']+[str(v) for v in fields])+'},\n')
    chunks.append('};\n')
    (output/'model_data.c').write_text(''.join(chunks),encoding='utf-8')
    return model_hash,metadata,sum(m['weight_payload_bytes'] for m in metadata)+fc['weights'].size


def command(arguments,log):
    completed=subprocess.run([str(a) for a in arguments],cwd=ROOT,capture_output=True,text=True)
    log.append(dict(command=[str(a) for a in arguments],returncode=completed.returncode,
                    stdout=completed.stdout,stderr=completed.stderr))
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)


def build_images(output,zig):
    logs=[]
    try:
        command([zig,'cc','-shared','-O2','-Wall','-Wextra','-Werror','-I',FW,
                 FW/'inference.c',output/'model_data.c','-o',output/'host_model.dll'],logs)
        command([zig,'cc','-target','thumb-freestanding-eabi','-mcpu=cortex_m3','-Os','-g',
                 '-ffreestanding','-fno-builtin','-ffunction-sections','-fdata-sections','-nostdlib',
                 '-I',FW,FW/'startup.S',FW/'board.c',FW/'inference.c',output/'model_data.c',
                 '-Wl,--gc-sections','-Wl,--build-id=none','-Wl,--entry=Reset_Handler','-Wl,-T,'+str(FW/'stm32f103c8.ld'),
                 '-o',output/'firmware.elf'],logs)
    finally:
        (output/'build_log.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
    from elftools.elf.elffile import ELFFile
    with (output/'firmware.elf').open('rb') as stream:
        elf=ELFFile(stream)
        (output/'firmware.map').write_text('ELF section map generated after linking by pyelftools\n'+
            '\n'.join(f"{s.name:24s} address=0x{s['sh_addr']:08x} bytes={s['sh_size']}" for s in elf.iter_sections()),encoding='utf-8')
        segments=[(s['p_paddr'],s.data()) for s in elf.iter_segments()
                  if s['p_type']=='PT_LOAD' and s['p_filesz'] and 0x08000000 <= s['p_paddr'] < 0x08010000]
        if not segments: raise ValueError('no Flash load segments')
        end=max(address+len(data) for address,data in segments)
        if end > 0x08010000: raise ValueError('Flash image exceeds C8 capacity')
        image=bytearray(b'\xff'*(end-0x08000000))
        for address,data in segments: image[address-0x08000000:address-0x08000000+len(data)]=data
        (output/'firmware.bin').write_bytes(image)
        symbols={s.name:s['st_value'] for s in elf.get_section_by_name('.symtab').iter_symbols()}
        if elf['e_entry'] != (symbols['Reset_Handler'] | 1):
            raise ValueError('ELF entry does not select the Thumb Reset_Handler')
        static=symbols['_end']-0x20000000
        if static+2048 > 20480: raise ValueError('less than 2 KiB stack headroom')
        if symbols['_flash_end']-0x08000000 != len(image): raise ValueError('reported Flash end differs from binary')
        return dict(flash_image_bytes=len(image),static_ram_bytes=static,
                    remaining_stack_space_bytes=20480-static,measured_mcu_peak_sram=False,
                    measured_mcu_latency=False,firmware_sha256=digest(output/'firmware.bin'))


def verify_host(output,layers,fc,inputs):
    library=ctypes.CDLL(str(output/'host_model.dll'))
    function=library.mcu_infer
    function.argtypes=[ctypes.POINTER(ctypes.c_uint8),ctypes.POINTER(ctypes.c_int32)]
    function.restype=None
    # Include extremes independently of natural data to check affine zero padding.
    cases=list(inputs)+[np.zeros((49,10),dtype=np.uint8),np.full((49,10),255,dtype=np.uint8),
        np.random.default_rng(20261101).integers(0,256,(49,10),dtype=np.uint8)]
    expected=[]
    for index,values in enumerate(cases):
        values=np.ascontiguousarray(values,dtype=np.uint8)
        reference=mcu.reference_infer(values,layers,fc)
        result=np.empty(12,dtype=np.int32)
        function(values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),result.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if not np.array_equal(reference,result):
            raise AssertionError(f'case {index}: C/NumPy logits differ: {result} vs {reference}')
        expected.append(result)
    return np.stack(expected[:len(inputs)]),dict(cases=len(cases),logits_per_case=12,
        max_abs_integer_difference=0,scope='host C versus independent NumPy integer reference; no MCU connected')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zig',type=Path,default=ROOT/'third_party/mcu_tools/zig-x86_64-windows-0.14.1/zig.exe')
    parser.add_argument('--masks',nargs='+',type=int)
    parser.add_argument('--smoke-samples',type=int,default=8)
    args=parser.parse_args()
    if not args.zig.is_file(): parser.error('provide an installed Zig 0.14.1 executable with --zig')
    compiler_version=subprocess.check_output([str(args.zig),'version'],text=True).strip()
    if compiler_version!='0.14.1': parser.error('this reproducible build pins Zig 0.14.1')
    if not 1 <= args.smoke_samples <= 64: parser.error('smoke-samples must be 1..64')
    run=ROOT/'results/round2/speech/seed42'
    selection=json.loads((run/'frozen_decisions.json').read_text(encoding='utf-8'))['budgets'][0]['policies']
    policy_masks={'uniform_w8':0,'archived_task':selection['raw_nll'],
                  'archived_cal':selection['mix_0.5'],'archived_ece':selection['ece'],'candidate_w4':127}
    masks=sorted(set(policy_masks.values())) if args.masks is None else sorted(set(args.masks))
    if any(not 0 <= mask < 128 for mask in masks): parser.error('mask must be 0..127')
    with np.load(ROOT/'results/round2/speech/features/quantization.npz',allow_pickle=False) as archive:
        features,labels=archive['features'],archive['labels']
    indices=np.linspace(0,len(features)-1,args.smoke_samples,dtype=int)
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    reports=[]
    for mask in masks:
        print(f'export mask={mask}',flush=True)
        output=ROOT/f'results/stm32f103/seed42_mask{mask}'
        output.mkdir(parents=True,exist_ok=True)
        layers,fc,scale,zero=export_model(checkpoint,features,mask,device)
        model_hash,metadata,payload=emit_data(output,layers,fc,mask,scale,zero)
        build=build_images(output,args.zig.resolve())
        inputs=mcu.quantize_inputs(features[indices],scale,zero)
        expected,verification=verify_host(output,layers,fc,inputs)
        np.savez_compressed(output/'golden.npz',inputs=inputs,expected_logits_q10=expected,
                            labels=labels[indices],role_indices=indices)
        manifest=dict(protocol='stm32f103_aux_integer_v1',seed=42,mask=mask,model_sha256=model_hash,
            aliases=[name for name,value in policy_masks.items() if value==mask],
            source_checkpoint_sha256=digest(run/'model.pt'),source_frozen_decisions_sha256=digest(run/'frozen_decisions.json'),
            source_policy_budget=1.0,source_policy_backend='historical float fake quantization; not reselected on deployment backend',
            quantization_role_sha256=digest(ROOT/'results/round2/speech/features/quantization.npz'),
            golden_sha256=digest(output/'golden.npz'),input_scale=scale,input_zero=zero,logits_scale=1/1024,
            shape=[49,10],layers=metadata,packed_weight_bytes=payload,range_calibration_samples=len(features),
            class_names=checkpoint['classes'],smoke_role='quantization',smoke_samples=len(inputs),
            numeric_contract='folded BN; signed per-channel W4/W8; unsigned affine A8; int32 accumulators; half-away integer requantization; integer global average; W8 classifier Q10 logits',
            hardware_status='not_connected_not_executed',temperature_fitted=False,
            compiler_version=compiler_version,
            source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in
                [*sorted(FW.glob('*.c')),*sorted(FW.glob('*.h')),*sorted(FW.glob('*.S')),
                 *sorted(FW.glob('*.ld')),ROOT/'cig_mpq/mcu.py',Path(__file__).resolve()]},
            build=build,host_verification=verification)
        (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        reports.append(manifest)
        print(json.dumps(dict(mask=mask,**build,host_cases=verification['cases'])),flush=True)
    (ROOT/'results/stm32f103/preparation_report.json').write_text(json.dumps(dict(configurations=reports,
        hardware_status='not_connected_not_executed',no_mcu_results_added_to_manuscript=True),ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__': main()
