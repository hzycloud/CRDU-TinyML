"""Independent edge checks for the STM32 integer contract and UART framing."""
import struct
import unittest
import zlib
import tempfile
from pathlib import Path
import numpy as np
from cig_mpq import mcu


class MCUTest(unittest.TestCase):
    def test_signed_int4_and_odd_tail(self):
        values = np.arange(-8, 8, dtype=np.int8)
        packed = mcu.pack_int4(values)
        self.assertEqual(packed[:2], bytes([0x98, 0xBA]))
        np.testing.assert_array_equal(mcu.unpack_int4(packed,16),values)
        self.assertEqual(mcu.pack_int4(np.array([-1,7,-8])), bytes([0x7F,0x08]))
        with self.assertRaises(ValueError): mcu.pack_int4(np.array([8]))

    def test_requantization_has_symmetric_half_away_rounding(self):
        np.testing.assert_array_equal(mcu.requantize(np.array([-5,-3,-1,0,1,3,5]),1,1),[-3,-2,-1,0,1,2,3])
        for value in (0.0001,0.25,1.0,8.0):
            mult, shift = mcu.multiplier(value)
            self.assertLess(abs(mult/(2.0**shift)-value),value*1e-8)

    def test_affine_padding_is_real_zero_and_odd_pool_drops_edge(self):
        x = np.full((3,3,1),121,dtype=np.uint8)
        layer = dict(weights=np.ones((1,3,3,1),dtype=np.int8),bias=np.array([2]),
                     mult=np.array([1]),shift=np.array([0]),in_zero=121,out_zero=10,
                     depthwise=False,relu=False,pool=1)
        np.testing.assert_array_equal(mcu.reference_layer(x,layer),np.full((3,3,1),12))
        layer.update(weights=np.ones((1,1,1,1),dtype=np.int8),in_zero=0,out_zero=0,
                     bias=np.array([0]),pool=2)
        x=np.arange(9,dtype=np.uint8).reshape(3,3,1)
        np.testing.assert_array_equal(mcu.reference_layer(x,layer),np.array([[[4]]]))

    def test_uart_request_and_response_reject_corruption(self):
        request=mcu.request_frame(17,np.zeros(490,dtype=np.uint8))
        self.assertEqual(len(request),502)
        self.assertEqual(zlib.crc32(request[:-4]),struct.unpack('<I',request[-4:])[0])
        values=[34,72000000,1234,16000,900,44000,0x410,64]
        body=struct.pack('<4sII8I32s12i',b'MRS1',17,2,*values,b'x'*32,*range(12))
        frame=body+struct.pack('<I',zlib.crc32(body))
        parsed=mcu.parse_response(frame,17,'78'*32)
        self.assertEqual(parsed['cycles'],1234)
        self.assertEqual(parsed['logits_q10'],list(range(12)))
        for bad in (frame[:-1],frame[:-1]+bytes([frame[-1]^1])):
            with self.assertRaises(ValueError): mcu.parse_response(bad,17,'78'*32)
        with self.assertRaises(ValueError): mcu.parse_response(frame,18,'78'*32)
        with self.assertRaises(ValueError): mcu.parse_response(frame,17,'00'*32)
        with self.assertRaises(ValueError): mcu.request_frame(0,np.zeros(489,dtype=np.uint8))

    def test_serial_reader_reassembles_fragmented_response(self):
        from scripts.capture_stm32 import read_response
        class FragmentedPort:
            def __init__(self): self.data=bytearray(b'boot-noise'+b'MRS1'+bytes(124))
            def read(self,count):
                count=min(count,3,len(self.data))
                answer=bytes(self.data[:count])
                del self.data[:count]
                return answer
        self.assertEqual(read_response(FragmentedPort(),1),b'MRS1'+bytes(124))

    def test_affine_input_rejects_nonfinite_values(self):
        with self.assertRaises(ValueError): mcu.quantize_inputs(np.array([np.nan]),1,0)
        np.testing.assert_array_equal(mcu.quantize_inputs(np.array([-100,0,100]),.5,120),[0,120,255])

    def test_depthwise_without_bias_exports_zero_integer_bias(self):
        from scripts.prepare_stm32 import fixed_constants
        bias,mult,shift=fixed_constants(None,.01,np.array([.02,.03]),.04,9,8)
        np.testing.assert_array_equal(bias,[0,0])
        self.assertTrue(np.all(mult>0))

    def test_saturated_stack_watermark_is_rejected(self):
        body=struct.pack('<4sII8I32s12i',b'MRS1',1,2,0,72000000,100,16536,3944,42360,0x410,64,b'x'*32,*([0]*12))
        frame=body+struct.pack('<I',zlib.crc32(body))
        with self.assertRaises(ValueError): mcu.parse_response(frame,1,'78'*32)

    def test_flash_readback_must_match_exact_firmware(self):
        from scripts.capture_stm32 import verify_flash_readback
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'readback.bin'
            path.write_bytes(b'older-build')
            with self.assertRaises(ValueError): verify_flash_readback(path,hashlib.sha256(b'new-build').hexdigest())
            self.assertEqual(verify_flash_readback(path,hashlib.sha256(b'older-build').hexdigest()),hashlib.sha256(b'older-build').hexdigest())

    def test_f103_erratum_uses_verified_swd_identity(self):
        body=struct.pack('<4sII8I32s12i',b'MRS1',1,1,34,72000000,0,16536,68,41496,0x307,64,b'x'*32,*([0]*12))
        frame=body+struct.pack('<I',zlib.crc32(body))
        with self.assertRaises(ValueError): mcu.parse_response(frame,1,'78'*32)
        debug=dict(device_id=0x20036410,flash_capacity_kib=64,cpuid=0x411FC231)
        self.assertEqual(mcu.parse_response(frame,1,'78'*32,debug_identity=debug)['identity_source'],'SWD_debug_access')
        with self.assertRaises(ValueError): mcu.parse_response(frame,1,'78'*32,debug_identity={**debug,'device_id':0x123})


if __name__=='__main__': unittest.main()
