"""Capture actual STM32 smoke inference and compare with preverified C logits.

Example: python scripts/capture_stm32.py --port COM5 --model-dir results/stm32f103/seed42_mask34
Smoke fixtures are quantization examples, not a test-set accuracy estimate.
"""
import argparse
import hashlib
import json
import re
import struct
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from cig_mpq.mcu import parse_response,request_frame


def verify_flash_readback(path,expected_sha256):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual!=expected_sha256:
        raise ValueError('actual MCU Flash does not match the exact local firmware binary')
    return actual


def readback_firmware(openocd,output,image_bytes,expected_sha256):
    path=(output/'flash_readback.bin').resolve()
    arguments=[str(openocd),'-f','interface/cmsis-dap.cfg','-c','transport select swd',
        '-f','target/stm32f1x.cfg','-c','adapter speed 1000','-c','init','-c','reset halt',
        '-c',f'dump_image {{{path.as_posix()}}} 0x08000000 {image_bytes}',
        '-c','mdw 0xe0042000 1','-c','mdh 0x1ffff7e0 1','-c','mdw 0xe000ed00 1',
        '-c','reset run','-c','shutdown']
    result=subprocess.run(arguments,capture_output=True,text=True,timeout=40)
    (output/'flash_readback.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode: raise RuntimeError('MCU Flash readback failed; see flash_readback.log')
    checksum=verify_flash_readback(path,expected_sha256)
    log=result.stdout+result.stderr
    def register(address):
        match=re.search(address+r':\s+([0-9a-f]+)',log,re.IGNORECASE)
        if not match: raise ValueError('SWD identity register missing from readback log')
        return int(match[1],16)
    identity=dict(sha256=checksum,device_id=register('0xe0042000'),
        flash_capacity_kib=register('0x1ffff7e0'),cpuid=register('0xe000ed00'))
    if identity['device_id']&0xFFF!=0x410 or ((identity['cpuid']>>4)&0xFFF)!=0xC23 or identity['flash_capacity_kib']<64:
        raise ValueError('SWD identity is not a supported F103 medium-density Cortex-M3')
    return identity


def read_response(port,timeout):
    deadline=time.monotonic()+timeout
    prefix=bytearray()
    while time.monotonic()<deadline:
        byte=port.read(1)
        if not byte: continue
        prefix.extend(byte)
        prefix=prefix[-4:]
        if prefix==b'MRS1': break
    else: raise TimeoutError('no STM32 response magic; check UART, power, reset and flashed image')
    frame=bytearray(prefix)
    while len(frame)<128 and time.monotonic()<deadline:
        frame.extend(port.read(128-len(frame)))
    if len(frame)!=128: raise TimeoutError('incomplete STM32 response')
    return bytes(frame)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=30)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--openocd',type=Path,default=ROOT/'third_party/mcu_tools/xpack-openocd-0.12.0-7/bin/openocd.exe')
    args=parser.parse_args()
    if not 1<=args.repeat<=100 or not 1<=args.timeout<=120: parser.error('repeat 1..100; timeout 1..120 seconds')
    folder=args.model_dir.resolve()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((folder/'golden.npz').read_bytes()).hexdigest()!=manifest['golden_sha256']:
        raise ValueError('golden fixture hash mismatch')
    if hashlib.sha256((folder/'firmware.bin').read_bytes()).hexdigest()!=manifest['build']['firmware_sha256']:
        raise ValueError('firmware binary hash mismatch')
    with np.load(folder/'golden.npz',allow_pickle=False) as archive:
        inputs,expected,indices=archive['inputs'],archive['expected_logits_q10'],archive['role_indices']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output or ROOT/f'results/stm32f103/hardware/mask{manifest["mask"]}_{stamp}'
    output.mkdir(parents=True,exist_ok=False)
    (output/'source_manifest.json').write_bytes((folder/'manifest.json').read_bytes())
    (output/'golden.npz').write_bytes((folder/'golden.npz').read_bytes())
    report=dict(started_at_utc=stamp,model_sha256=manifest['model_sha256'],mask=manifest['mask'],
                firmware_sha256=manifest['build']['firmware_sha256'],port=args.port,
                fixture_role='quantization_smoke_not_test_accuracy',state='started',records=[],
                limitations=['Clock frequency is firmware configuration, not externally measured oscillator frequency.',
                    'Observed stack watermark is workload-specific, not a proof of all possible execution paths.',
                    'STM32F103 ES096 2.2.2: runtime DBGMCU ID is informational; identity is verified through SWD.',
                    'No energy measurement; input MFCC features are prepared on the PC.'])
    trace=[]
    try:
        report['debug_identity']=readback_firmware(args.openocd,output,
            manifest['build']['flash_image_bytes'],manifest['build']['firmware_sha256'])
        report['flash_readback_sha256']=report['debug_identity']['sha256']
        import serial
        with serial.Serial(args.port,115200,timeout=.2,write_timeout=2) as port:
            time.sleep(1)
            port.reset_input_buffer()
            sequence=0
            def exchange(values=None):
                nonlocal sequence
                sequence+=1
                request=request_frame(sequence,values)
                port.write(request)
                port.flush()
                raw=read_response(port,args.timeout)
                trace.append(dict(request_hex=request.hex(),response_hex=raw.hex()))
                result=parse_response(raw,sequence,manifest['model_sha256'],debug_identity=report['debug_identity'])
                if result['status']!=(1 if values is None else 2) or result['mask']!=manifest['mask']:
                    raise ValueError('wrong response kind or mask')
                for key in ('flash_image_bytes','static_ram_bytes'):
                    if result[key]!=manifest['build'][key]: raise ValueError(f'firmware {key} differs from compiled manifest')
                return result
            report['info']=exchange()
            report['warmup']=exchange(inputs[0])
            for repeat in range(args.repeat):
                for index,values in enumerate(inputs):
                    result=exchange(values)
                    result.update(repeat=repeat,smoke_index=index,quantization_role_index=int(indices[index]))
                    result['matches_host_reference']=bool(np.array_equal(result['logits_q10'],expected[index]))
                    report['records'].append(result)
                    if not result['matches_host_reference']: raise ValueError(f'board logits differ from host at example {index}')
                    if result['cycles']==0: raise ValueError('DWT cycle counter returned zero; no latency accepted')
                    print(f"mask={manifest['mask']} sample={index} cycles={result['cycles']} matched",flush=True)
            times=np.array([r['cycles']/r['clock_hz']*1000 for r in report['records']])
            report.update(state='onboard_smoke_passed',all_logits_match=True,
                measured_inferences=len(times),nominal_clock_latency_ms=dict(mean=float(times.mean()),
                median=float(np.median(times)),p95=float(np.quantile(times,.95))),
                max_observed_stack_bytes=max(r['stack_peak_bytes'] for r in report['records']))
    except Exception as error:
        report.update(state='failed',error=repr(error))
        raise
    finally:
        (output/'capture.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'serial_frames.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')
        print(output,flush=True)


if __name__=='__main__': main()
