import struct


def parse_aria2_control(data: bytes) -> dict:
    version = struct.unpack(">H", data[:2])[0]
    offset = 2
    extension = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    info_hash_length = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    if extension not in {0, 1}:
        raise ValueError(f"unsupported aria2 control extension: {extension}")
    offset += info_hash_length
    piece_length = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    total_length = struct.unpack(">Q", data[offset : offset + 8])[0]
    offset += 8
    upload_length = struct.unpack(">Q", data[offset : offset + 8])[0]
    offset += 8
    bitfield_length = struct.unpack(">I", data[offset : offset + 4])[0]
    offset += 4
    bitfield = data[offset : offset + bitfield_length]
    if len(bitfield) != bitfield_length:
        raise ValueError("truncated aria2 control bitfield")

    piece_count = (total_length + piece_length - 1) // piece_length
    completed = []
    missing = []
    for index in range(piece_count):
        byte = bitfield[index // 8]
        bit = (byte >> (7 - index % 8)) & 1
        (completed if bit else missing).append(index)
    return {
        "version": version,
        "extension": extension,
        "piece_length": piece_length,
        "total_length": total_length,
        "upload_length": upload_length,
        "piece_count": piece_count,
        "completed_pieces": completed,
        "missing_pieces": missing,
        "bitfield_length": bitfield_length,
    }
