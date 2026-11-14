"""Local, pickle-free request/response wire format for native inference."""

import io
import struct

import numpy as np

MAX_PAYLOAD = 16 * 1024 * 1024


def send(connection, **values):
    buffer = io.BytesIO()
    np.savez(buffer, **values)
    payload = buffer.getvalue()
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Oversized inference message")
    connection.sendall(struct.pack("!Q", len(payload)) + payload)


def receive(connection):
    def exact(length):
        parts = []
        while length:
            part = connection.recv(length)
            if not part:
                raise EOFError("Native inference connection closed")
            parts.append(part)
            length -= len(part)
        return b"".join(parts)

    length = struct.unpack("!Q", exact(8))[0]
    if length > MAX_PAYLOAD:
        raise ValueError("Oversized inference message")
    with np.load(io.BytesIO(exact(length)), allow_pickle=False) as payload:
        return {key: payload[key].copy() for key in payload.files}
