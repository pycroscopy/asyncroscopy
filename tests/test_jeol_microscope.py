import numpy as np

from asyncroscopy.instruments.electron_microscope.jeol import JeolMicroscope


class TestDecodeRawdata:
    def test_decodes_bytes_into_2d_int16_image(self) -> None:
        # PyJEM's snapshot_rawdata() returns a raw little-endian int16 byte buffer,
        # not an array — the decoder must turn it into a 2D image.
        original = np.arange(6, dtype=np.dtype('<i2')).reshape((2, 3))

        decoded = JeolMicroscope._decode_rawdata(original.tobytes(), width=2, height=3)

        assert isinstance(decoded, np.ndarray)
        assert decoded.dtype == np.dtype('<i2')
        assert decoded.shape == (2, 3)
        assert decoded.tolist() == original.tolist()

    def test_round_trips_a_square_frame(self) -> None:
        original = (np.arange(512 * 512, dtype=np.dtype('<i2')) % 1000).reshape((512, 512))

        decoded = JeolMicroscope._decode_rawdata(original.tobytes(), width=512, height=512)

        assert np.array_equal(decoded, original)
