"""Separate, fixed-point deployment contract for the STM32 auxiliary experiment.

This backend folds BN and quantizes inputs/pool outputs. It is deliberately not
claimed equivalent to the paper's float fake-quantization operator protocol.
"""
import math
import struct
import zlib
import numpy as np


def pack_int4(values):
    values = np.asarray(values).reshape(-1)
    if not np.isfinite(values).all() or np.any(values != np.rint(values)) or np.any((values < -8) | (values > 7)):
        raise ValueError('signed INT4 weights must be integers in [-8,7]')
    nibbles = (values.astype(np.int8).astype(np.uint8) & 15)
    if len(nibbles) % 2:
        nibbles = np.append(nibbles,np.uint8(0))
    return (nibbles[::2] | (nibbles[1::2] << 4)).tobytes()


def unpack_int4(payload,count):
    if count < 0 or len(payload) != (count+1)//2:
        raise ValueError('INT4 payload length does not match weight count')
    packed = np.frombuffer(payload,dtype=np.uint8)
    values = np.empty(len(packed)*2,dtype=np.int8)
    for offset,nibbles in ((0,packed & 15),(1,packed >> 4)):
        values[offset::2] = np.where(nibbles >= 8,nibbles.astype(np.int16)-16,nibbles)
    return values[:count]


def multiplier(ratio):
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError('requantization ratio must be finite and positive')
    fraction,exponent = math.frexp(float(ratio))
    value = round(fraction * (1 << 31))
    if value == 1 << 31:
        value //= 2
        exponent += 1
    shift = 31-exponent
    if not 0 <= shift <= 62:
        raise ValueError('requantization ratio is outside supported fixed-point range')
    return value,shift


def requantize(values,mult,shift):
    """Integer multiplication followed by symmetric half-away-from-zero rounding."""
    values = np.asarray(values,dtype=np.int64)
    mult,shift = np.asarray(mult,dtype=np.int64),np.asarray(shift,dtype=np.int64)
    if np.any((shift < 0) | (shift > 62)):
        raise ValueError('invalid right shift')
    product = values * mult
    rounding = np.where(shift > 0,np.left_shift(np.int64(1),np.maximum(shift-1,0)),0)
    return np.sign(product) * np.right_shift(np.abs(product)+rounding,shift)


def affine_parameters(low,high):
    low,high = min(float(low),0.),max(float(high),0.)
    scale = max((high-low)/255,1e-12)
    zero = int(np.clip(np.rint(-low/scale),0,255))
    return scale,zero


def quantize_inputs(values,scale,zero):
    if not np.isfinite(values).all() or not math.isfinite(scale) or scale <= 0 or not 0 <= zero <= 255:
        raise ValueError('invalid affine input')
    return np.clip(np.rint(np.asarray(values,dtype=np.float64)/scale)+zero,0,255).astype(np.uint8)


def reference_layer(inputs,layer):
    """NumPy integer reference; HWC activations and OHWI weights."""
    inputs = np.asarray(inputs,dtype=np.uint8)
    weight = np.asarray(layer['weights'],dtype=np.int64)
    kernel = weight.shape[1]
    height,width,channels = inputs.shape
    padded = np.pad(inputs.astype(np.int64),((kernel//2,kernel//2),(kernel//2,kernel//2),(0,0)),
                    constant_values=layer['in_zero']) - layer['in_zero']
    patches = np.lib.stride_tricks.sliding_window_view(padded,(kernel,kernel),axis=(0,1)).transpose(0,1,3,4,2)
    if layer['depthwise']:
        accumulators = np.einsum('hwklc,ckl->hwc',patches,weight[:,:,:,0],optimize=True)
    else:
        accumulators = np.tensordot(patches,weight,axes=([2,3,4],[1,2,3]))
    values = requantize(accumulators+layer['bias'],layer['mult'],layer['shift'])+layer['out_zero']
    values = np.clip(values,layer['out_zero'] if layer['relu'] else 0,255).astype(np.uint8)
    pool = layer['pool']
    if pool > 1:
        height,width = height//pool,width//pool
        values = values[:height*pool,:width*pool].reshape(height,pool,width,pool,-1).max(axis=(1,3))
    return values


def reference_infer(inputs,layers,fc):
    current = np.asarray(inputs,dtype=np.uint8).reshape(49,10,1)
    for layer in layers:
        current = reference_layer(current,layer)
    cells = current.shape[0]*current.shape[1]
    pooled = (current.astype(np.int64).sum(axis=(0,1))+cells//2)//cells
    accumulators = (pooled-fc['in_zero']) @ fc['weights'].astype(np.int64).T + fc['bias']
    return requantize(accumulators,fc['mult'],fc['shift']).astype(np.int32)


def request_frame(sequence,inputs=None):
    if not 0 <= sequence < 2**32:
        raise ValueError('sequence must be uint32')
    payload = b''
    if inputs is not None:
        inputs = np.asarray(inputs)
        if inputs.dtype != np.uint8 or inputs.size != 490:
            raise ValueError('request needs exactly 490 uint8 input features')
        payload = inputs.tobytes()
    body = (b'MQI1' if inputs is None else b'MQR1')+struct.pack('<I',sequence)+payload
    return body+struct.pack('<I',zlib.crc32(body))


def parse_response(frame,sequence,model_hash,debug_identity=None):
    if len(frame) != 128 or zlib.crc32(frame[:-4]) != struct.unpack('<I',frame[-4:])[0]:
        raise ValueError('incomplete response or CRC mismatch')
    magic,actual,status,*fields = struct.unpack('<4sII8I32s12i',frame[:-4])
    if magic != b'MRS1' or actual != sequence or status not in (1,2):
        raise ValueError('invalid response magic, sequence or device status')
    if fields[8].hex() != model_hash:
        raise ValueError('response model hash differs from fixture')
    names = ('mask','clock_hz','cycles','static_ram_bytes','stack_peak_bytes',
             'flash_image_bytes','device_id','flash_capacity_kib')
    result = dict(zip(names,fields[:8]))
    if result['clock_hz'] not in (8000000,72000000):
        raise ValueError('unsupported reported clock')
    # ST ES096 section 2.2.2: user software cannot reliably read DBGMCU.
    # A capture with that erratum must validate identity via the SWD readback.
    identity=result if debug_identity is None else debug_identity
    if (identity['device_id'] & 0xFFF) != 0x410 or identity['flash_capacity_kib'] < 64:
        raise ValueError('device identity/capacity does not match STM32F103 medium-density target')
    if debug_identity is not None:
        if ((identity['cpuid'] >> 4) & 0xFFF)!=0xC23 or identity['flash_capacity_kib']!=result['flash_capacity_kib']:
            raise ValueError('SWD core/capacity differs from expected target telemetry')
        result.update(identity_source='SWD_debug_access',debug_device_id=identity['device_id'],cpuid=identity['cpuid'])
    if result['static_ram_bytes']+result['stack_peak_bytes'] >= 20480 or result['flash_image_bytes'] > 65536:
        raise ValueError('reported resource use exceeds target limits')
    result.update(status=status,model_sha256=fields[8].hex(),logits_q10=list(fields[9:]))
    return result
