"""Audit saved physical UART frames and summarize bounded MCU smoke evidence."""
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from cig_mpq.mcu import parse_response


def main():
    base=ROOT/'results/stm32f103'
    valid=[]
    failed=[]
    for path in sorted((base/'hardware').glob('*/capture.json')):
        capture=json.loads(path.read_text(encoding='utf-8'))
        if capture['state']!='onboard_smoke_passed':
            failed.append(dict(folder=path.parent.name,state=capture['state'],error=capture.get('error')))
            continue
        folder=path.parent
        model=base/f'seed42_mask{capture["mask"]}'
        manifest_path=folder/'source_manifest.json'
        snapshot=manifest_path.exists()
        manifest=json.loads((manifest_path if snapshot else model/'manifest.json').read_text(encoding='utf-8'))
        golden=folder/'golden.npz'
        with np.load(golden if golden.exists() else model/'golden.npz',allow_pickle=False) as archive:
            expected={x.tobytes():y for x,y in zip(archive['inputs'],archive['expected_logits_q10'])}
        actual=hashlib.sha256((folder/'flash_readback.bin').read_bytes()).hexdigest()
        assert actual==capture['firmware_sha256']==manifest['build']['firmware_sha256']
        trace=json.loads((folder/'serial_frames.json').read_text(encoding='utf-8'))
        assert len(trace)==len(capture['records'])+2
        run_responses=[]
        for sequence,row in enumerate(trace,1):
            request,response=bytes.fromhex(row['request_hex']),bytes.fromhex(row['response_hex'])
            assert zlib.crc32(request[:-4])==struct.unpack('<I',request[-4:])[0]
            assert struct.unpack('<I',request[4:8])[0]==sequence
            result=parse_response(response,sequence,manifest['model_sha256'],capture['debug_identity'])
            if request[:4]==b'MQI1':
                assert len(request)==12 and result['status']==1
            else:
                assert request[:4]==b'MQR1' and len(request)==502 and result['status']==2
                np.testing.assert_array_equal(result['logits_q10'],expected[request[8:-4]])
                run_responses.append(result)
        assert len(run_responses)==len(capture['records'])+1
        for raw,record in zip(run_responses[1:],capture['records']):
            for key in ('cycles','clock_hz','stack_peak_bytes','logits_q10'):
                assert raw[key]==record[key]
        milliseconds=np.array([r['cycles']/r['clock_hz']*1000 for r in capture['records']])
        valid.append(dict(mask=capture['mask'],aliases=manifest['aliases'],folder=folder.relative_to(base).as_posix(),
            model_sha256=manifest['model_sha256'],firmware_sha256=actual,
            measured_inferences=len(milliseconds),warmup_inferences=1,unique_quantization_examples=len(expected),
            all_12_logits_exact=True,all_raw_frame_crc_verified=True,flash_readback_verified=True,
            source_snapshot_saved_at_capture=snapshot,clock_hz=capture['info']['clock_hz'],
            flash_image_bytes=capture['info']['flash_image_bytes'],static_ram_bytes=capture['info']['static_ram_bytes'],
            max_observed_stack_bytes=capture['max_observed_stack_bytes'],
            observed_sram_bytes=capture['info']['static_ram_bytes']+capture['max_observed_stack_bytes'],
            latency_ms=dict(mean=float(milliseconds.mean()),median=float(np.median(milliseconds)),
                p95=float(np.quantile(milliseconds,.95))),debug_identity=capture['debug_identity']))
    order={0:0,34:1,2:2,35:3,127:4}
    valid.sort(key=lambda row:order.get(row['mask'],row['mask']+5))
    report=dict(protocol='STM32F103 source-disjoint-speech auxiliary integer smoke v1',
        valid_runs=valid,failed_runs_retained=failed,
        total_measured_inferences=sum(v['measured_inferences'] for v in valid),
        limitations=['One physical board, one training seed, eight quantization-role examples.',
            'Repeated examples characterize engineering timing and parity, not independent predictive performance.',
            'Nominal firmware clock, no externally measured oscillator frequency; PC computes MFCC.',
            'Folded-BN fixed-point backend differs from original fake quantization; policies not reoptimized.',
            'No test-set accuracy/NLL/ECE, mixed-precision utility, energy or multi-board generalization claim.'])
    (base/'hardware_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# STM32F103C8T6 实际上板工程验证（2026-11-01）','',
        '本报告来自实际DAP烧录、Flash回读与COM4串口返回。原始失败记录保留，不计为成功结果。','',
        '| Mask | 配置来源 | Flash镜像 bytes | 静态RAM bytes | 观测栈 bytes | 观测SRAM bytes | 平均网络推理 ms | 核对推理次数 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in valid:
        lines.append(f"| {row['mask']} | {', '.join(row['aliases'])} | {row['flash_image_bytes']} | {row['static_ram_bytes']} | {row['max_observed_stack_bytes']} | {row['observed_sram_bytes']} | {row['latency_ms']['mean']:.3f} | {row['measured_inferences']} |")
    lines+=['','全部成功运行的12个整数logits均与主机参考逐项相同；每个配置均在运行前回读Flash并核对固件SHA-256。',
        '身份经SWD验证：Cortex-M3，芯片类别0x410，标称Flash容量64 KiB。CPU运行态读取调试ID有ST ES096 2.2.2所述限制，未用它代替DAP身份。',
        '时钟列见hardware_summary.json；耗时由DWT周期除以固件报告的配置频率计算，不包含串口与电脑MFCC。',
        '观测SRAM=链接静态RAM+本工作负载的栈水位；不表示所有路径的数学最坏上界。',
        '当前通用标量C后端主要用于功能与资源核验；实测耗时见表，尚不能据此宣称实时语音或W4加速。',
        '', '## 证据边界','',*['- '+s for s in report['limitations']],
        '', '## 可加入论文的英文补充段落','',
        'We additionally performed an engineering deployment check of the seed-42 speech network on a physical STM32F103C8T6-class Cortex-M3 board with 64 KiB Flash and 20 KiB SRAM. A standalone integer backend used folded batch normalization, packed W4/W8 weights, affine A8 activations and integer requantization. Five archived configuration assignments were examined using eight quantization-role examples, each repeated three times after a warm-up. The executable Flash contents were read back and matched to the exact firmware SHA-256 before each capture. All twelve fixed-point logits agreed exactly with the host integer reference. DWT cycles and observed stack watermarks provide workload-specific implementation evidence. These checks do not estimate held-out accuracy or calibration, and do not establish equivalence with the original fake-quantization backend, a latency advantage from W4, or an energy benefit.',
        '', '2026-10-28论文与归档保留为历史版本；本报告是后续硬件证据，正式全文整合仍需保持上述范围。']
    (base/'STM32实际验证报告_2026-11-01.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(valid_masks=[v['mask'] for v in valid],measured_inferences=report['total_measured_inferences'],failed_runs_retained=len(failed)),ensure_ascii=False))


if __name__=='__main__': main()
