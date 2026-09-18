"""Austin is working on: Inactive MRC conversion reference pending the Tiled/SciFiReaders adapter.
worked under the old schema, now outdated. will use as an outline for remote MRC conversion and registration once the new schema is in place."""

# Former DATA method; not runnable without replacing its DATA state and registration.
# import json
# import os
# from datetime import datetime
# from pathlib import Path
#
# import h5py
# from SciFiReaders import MRCReader
#
# @command(dtype_in=str, dtype_out=str)
# def copy_and_register_remote_file(self, request_json: str) -> str:
#     """Convert a remote AutoScript MRC file to HDF5 and register it.
#
#     The source MRC is left untouched. The HDF5 file is written under the
#     configured data save path and becomes visible only after an atomic
#     rename, so Tiled can never observe a partial conversion.
#     """
#     request = json.loads(request_json)
#     source = Path(request["source_path"]).expanduser()
#     if source.suffix.lower() not in {".mrc", ".mrcs"}:
#         raise ValueError(f"Expected an MRC source file, received: {source}")
#     if not source.is_file():
#         raise FileNotFoundError(f"Remote MRC file is not readable: {source}")
#
#     destination_directory = Path(self._save_path).expanduser()
#     destination_directory.mkdir(parents=True, exist_ok=True)
#     detector = str(request.get("detector", "BM-Ceta"))
#     stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
#     destination = destination_directory / f"stem_data_{detector}_{stamp}.h5"
#     partial = destination.with_suffix(".h5.partial")
#
#     try:
#         channels = MRCReader(str(source)).read()
#         if "Channel_000" not in channels:
#             raise ValueError(f"SciFiReaders did not return Channel_000 for {source}")
#         source_data = channels["Channel_000"]
#         if len(source_data.shape) != 4:
#             raise ValueError(f"Expected SciFiReaders to return 4D-STEM data, received shape {source_data.shape}")
#
#         requested_scan_shape = tuple(int(value) for value in request.get("scan_shape", []))
#         if requested_scan_shape and tuple(source_data.shape[:2]) != requested_scan_shape:
#             raise ValueError(f"MRC scan shape {source_data.shape[:2]} does not match requested scan shape {requested_scan_shape}")
#
#         with h5py.File(partial, "w", track_order=True) as h5:
#             dataset = h5.create_dataset("stem_data", shape=source_data.shape, dtype=source_data.dtype, chunks=True)
#             for row in range(source_data.shape[0]):
#                 for column in range(source_data.shape[1]):
#                     frame = source_data[row, column]
#                     dataset[row, column] = frame.compute() if hasattr(frame, "compute") else frame
#
#             dataset.attrs["acquisition_type"] = "stem_data"
#             dataset.attrs["detector"] = str(request.get("detector", "BM-Ceta"))
#             dataset.attrs["source_format"] = "MRC"
#             dataset.attrs["source_file"] = str(source)
#             dataset.attrs["data_type"] = str(getattr(source_data, "data_type", "image_4d"))
#             for name in ("dwell_time", "scan_region", "scan_shape"):
#                 if name in request:
#                     value = request[name]
#                     dataset.attrs[name] = value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
#
#             h5.attrs["source_mrc_metadata_json"] = json.dumps(getattr(source_data, "original_metadata", {}), default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
#
#         os.replace(partial, destination)
#     except Exception:
#         partial.unlink(missing_ok=True)
#         raise
#
#     return self.register_acquisition_file(str(destination))
