import struct
import unittest

from cig_mpq.aria2_control import parse_aria2_control


class Aria2ControlTest(unittest.TestCase):
    def test_parse_control_file_identifies_missing_piece(self):
        data = (
            b"\x00\x01"
            + struct.pack(">I", 0)
            + struct.pack(">I", 0)
            + struct.pack(">I", 4)
            + struct.pack(">Q", 10)
            + struct.pack(">Q", 0)
            + struct.pack(">I", 1)
            + bytes([0b10100000])
        )

        result = parse_aria2_control(data)

        self.assertEqual(result["piece_length"], 4)
        self.assertEqual(result["total_length"], 10)
        self.assertEqual(result["piece_count"], 3)
        self.assertEqual(result["completed_pieces"], [0, 2])
        self.assertEqual(result["missing_pieces"], [1])


if __name__ == "__main__":
    unittest.main()
