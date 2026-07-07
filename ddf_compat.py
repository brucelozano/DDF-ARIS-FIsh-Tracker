#!/usr/bin/env python3
"""
DDF compatibility parser.

Provides version-dispatch parsing for DDF versions 0-5 and normalized helpers
for frame offset calculations, range extraction, and timestamp conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import struct
from typing import Any, Dict, Optional


MATLAB_EPOCH_OFFSET_DAYS = 719529.0  # MATLAB datenum for 1970-01-01 00:00:00

CLASSIC_S_WINDOWS = [1.125, 2.25, 4.5, 9.0, 18.0, 36.0]
EXTENDED_S_WINDOWS = [1.25, 2.5, 5.0, 10.0, 20.0, 40.0]
EXTENDED_LR_WINDOWS = [2.5, 5.0, 10.0, 20.0, 40.0, 80.0]
CLASSIC_LR_WINDOWS = [2.25, 4.5, 9.0, 18.0, 36.0, 72.0]

FRAME_HEADER_LENGTH_BY_VERSION = {
    0: 124,
    1: 124,
    2: 260,
    3: 256,
    4: 1024,
    5: 1024,
}


@dataclass
class DDFLayout:
    """Normalized binary layout and metadata for one DDF file."""

    magic: str
    version: int
    file_header_length: int
    frame_header_length: int
    numframes: int
    framerate: float
    resolution: int
    numbeams: int
    sampleperchannel: int
    samplerate: float
    receivergain: int
    reverse: int
    serialnumber: int
    configuration: str
    windowstart: float
    windowlength: float
    minrange: float
    maxrange: float

    def frame_data_length(self) -> int:
        return int(self.numbeams) * int(self.sampleperchannel)

    def frame_stride(self) -> int:
        return int(self.frame_header_length) + self.frame_data_length()

    def frame_header_offset(self, frame_number: int) -> int:
        if frame_number < 1:
            raise ValueError(f"Frame number must be >= 1, got {frame_number}")
        return int(self.file_header_length) + (int(frame_number) - 1) * self.frame_stride()

    def frame_data_offset(self, frame_number: int) -> int:
        return self.frame_header_offset(frame_number) + int(self.frame_header_length)

    def to_file_info(self) -> Dict[str, Any]:
        return {
            "version": int(self.version),
            "serialnumber": int(self.serialnumber),
            "configuration": str(self.configuration),
            "framerate": float(self.framerate),
            "minrange": float(self.minrange),
            "maxrange": float(self.maxrange),
            "numbeams": int(self.numbeams),
            "reverse": int(self.reverse),
            "fileheaderlength": int(self.file_header_length),
            "frameheaderlength": int(self.frame_header_length),
            "numframes": int(self.numframes),
            "sampleperchannel": int(self.sampleperchannel),
            "receivergain": int(self.receivergain),
            "resolution": int(self.resolution),
            "samplerate": float(self.samplerate),
            "windowstart": float(self.windowstart),
            "windowlength": float(self.windowlength),
        }


def read_file_layout(fid) -> DDFLayout:
    """Parse file-level metadata and return a normalized DDF layout object."""
    fid.seek(0)
    magic_bytes = _read_bytes(fid, 3)
    if magic_bytes != b"DDF":
        raise ValueError("Not a DDF file")

    version = _read_u8(fid)
    if version not in FILE_HEADER_PARSERS:
        raise ValueError(f"Unsupported DDF version: {version}")

    file_meta = FILE_HEADER_PARSERS[version](fid)
    file_header_length = int(fid.tell())
    frame_header_length = int(file_meta.get("frame_header_length", FRAME_HEADER_LENGTH_BY_VERSION[version]))

    layout = DDFLayout(
        magic=magic_bytes.decode("ascii", errors="ignore"),
        version=int(version),
        file_header_length=file_header_length,
        frame_header_length=frame_header_length,
        numframes=int(file_meta["numframes"]),
        framerate=float(file_meta["framerate"]),
        resolution=int(file_meta["resolution"]),
        numbeams=int(file_meta["numbeams"]),
        sampleperchannel=int(file_meta["sampleperchannel"]),
        samplerate=float(file_meta["samplerate"]),
        receivergain=int(file_meta.get("receivergain", 0)),
        reverse=int(file_meta.get("reverse", 0)),
        serialnumber=int(file_meta.get("serialnumber", 0)),
        configuration=str(file_meta.get("configuration", "Unknown")),
        windowstart=float(file_meta.get("windowstart", 0.0)),
        windowlength=float(file_meta.get("windowlength", 0.0)),
        minrange=float(file_meta.get("windowstart", 0.0)),
        maxrange=float(file_meta.get("windowstart", 0.0) + file_meta.get("windowlength", 0.0)),
    )

    # Initialize range/configuration from first frame header when available.
    if layout.numframes > 0:
        try:
            first_header = read_frame_header(fid, layout, layout.frame_header_offset(1))
            layout.windowstart = float(first_header.get("windowstart", layout.windowstart))
            layout.windowlength = float(first_header.get("windowlength", layout.windowlength))
            layout.minrange = float(layout.windowstart)
            layout.maxrange = float(layout.windowstart + layout.windowlength)
            if first_header.get("receivergain") is not None:
                layout.receivergain = int(first_header.get("receivergain", layout.receivergain))
            if first_header.get("configflags") is not None:
                layout.configuration = _configuration_from_flags(int(first_header["configflags"]))
        except Exception:
            # Keep file-header defaults if first-frame decode fails.
            pass

    fid.seek(layout.file_header_length)
    return layout


def read_frame_header(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    """Parse one frame header for the given layout/version and normalize fields."""
    parser = FRAME_HEADER_PARSERS.get(layout.version)
    if parser is None:
        raise ValueError(f"No frame-header parser for DDF version {layout.version}")
    return parser(fid, layout, frame_position)


def _parse_file_header_v5(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    numbeams = _read_i32(fid)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart = _read_f32(fid)
    windowlength = _read_f32(fid)
    reverse = _read_i32(fid)
    serialnumber = _read_i32(fid)

    # v5 file header is 1024 bytes total (including magic+version).
    remaining = 1024 - int(fid.tell())
    if remaining > 0:
        fid.read(remaining)

    if serialnumber >= 1000:
        configuration = "DIDSON-S Extended Windows"
    else:
        configuration = "DIDSON Standard"

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": reverse,
        "serialnumber": serialnumber,
        "configuration": configuration,
        "frame_header_length": 1024,
    }


def _parse_file_header_v4(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    numbeams = _read_i32(fid)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    reverse = _read_i32(fid)
    serialnumber = _read_i32(fid)

    fid.read(32)   # date
    fid.read(256)  # idstring
    _ = _read_i32(fid)  # id1
    _ = _read_i32(fid)  # id2
    _ = _read_i32(fid)  # id3
    _ = _read_i32(fid)  # id4
    _ = _read_i32(fid)  # startframe
    _ = _read_i32(fid)  # endframe
    _ = _read_i32(fid)  # timelapse
    _ = _read_i32(fid)  # recordInterval
    _ = _read_i32(fid)  # radioseconds
    _ = _read_i32(fid)  # frameinterval
    fid.read(648)       # userassigned padding

    # Convert using classic defaults as a baseline; frame headers can override.
    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, resolution, clamp_legacy=False
    )

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": reverse,
        "serialnumber": serialnumber,
        "configuration": "Unknown",
        "frame_header_length": 1024,
    }


def _parse_file_header_v3(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    numbeams = _read_i32(fid)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    reverse = _read_i32(fid)
    serialnumber = _read_i32(fid)

    fid.read(32)   # date
    fid.read(256)  # idstring
    _ = _read_i32(fid)  # id1
    _ = _read_i32(fid)  # id2
    _ = _read_i32(fid)  # id3
    _ = _read_i32(fid)  # id4
    _ = _read_i32(fid)  # startframe
    _ = _read_i32(fid)  # endframe
    _ = _read_i32(fid)  # timelapse
    _ = _read_i32(fid)  # recordInterval
    _ = _read_i32(fid)  # radioseconds
    _ = _read_i32(fid)  # frameinterval
    fid.read(136)       # userassigned

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, resolution, clamp_legacy=False
    )

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": reverse,
        "serialnumber": serialnumber,
        "configuration": "Unknown",
        "frame_header_length": 256,
    }


def _parse_file_header_v2(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    _numbeams_raw = _read_i32(fid)
    numbeams = 48 + 48 * (1 if resolution == 1 else 0)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    reverse = _read_i32(fid)
    serialnumber = _read_i32(fid)

    fid.read(32)   # date
    fid.read(256)  # idstring
    _ = _read_i32(fid)  # id1
    _ = _read_i32(fid)  # id2
    _ = _read_i32(fid)  # id3
    _ = _read_i32(fid)  # id4
    _ = _read_i32(fid)  # startframe
    _ = _read_i32(fid)  # endframe
    fid.read(152)       # userassigned

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, resolution, clamp_legacy=True
    )

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": reverse,
        "serialnumber": serialnumber,
        "configuration": _configuration_from_flags(1),
        "frame_header_length": 260,
    }


def _parse_file_header_v1(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    numbeams = _read_i32(fid)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    reverse = _read_i32(fid)
    serialnumber = _read_i32(fid)

    length1 = _read_u8(fid)
    fid.read(length1)
    length2 = _read_u8(fid)
    fid.read(length2)

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, resolution, clamp_legacy=True
    )

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": reverse,
        "serialnumber": serialnumber,
        "configuration": _configuration_from_flags(1),
        "frame_header_length": 124,
    }


def _parse_file_header_v0(fid) -> Dict[str, Any]:
    numframes = _read_i32(fid)
    framerate = _read_i32(fid)
    resolution = _read_i32(fid)
    numbeams = _read_i32(fid)
    samplerate = _read_f32(fid)
    sampleperchannel = _read_i32(fid)
    receivergain = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)

    length1 = _read_u8(fid)
    fid.read(length1)
    length2 = _read_u8(fid)
    fid.read(length2)

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, resolution, clamp_legacy=True
    )

    return {
        "numframes": numframes,
        "framerate": framerate,
        "resolution": resolution,
        "numbeams": numbeams,
        "samplerate": samplerate,
        "sampleperchannel": sampleperchannel,
        "receivergain": receivergain,
        "windowstart": windowstart,
        "windowlength": windowlength,
        "reverse": 0,
        "serialnumber": 0,
        "configuration": _configuration_from_flags(1),
        "frame_header_length": 124,
    }


def _parse_frame_header_v5(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    fid.seek(frame_position)
    framenumber = _read_u32(fid)
    frametime = _read_u64(fid)
    _version_marker = _read_u32(fid)
    status = _read_u32(fid)
    sonartimestamp = _read_u64(fid)
    tsday = _read_u32(fid)
    tshour = _read_u32(fid)
    tsminute = _read_u32(fid)
    tssecond = _read_u32(fid)
    tshsecond = _read_u32(fid)
    transmitmode = _read_u32(fid)
    windowstart = _read_f32(fid)
    windowlength = _read_f32(fid)
    threshold = _read_u32(fid)
    intensity = _read_i32(fid)
    receivergain = _read_u32(fid)
    degc1 = _read_u32(fid)
    degc2 = _read_u32(fid)
    humidity = _read_u32(fid)
    focus = _read_u32(fid)
    battery = _read_u32(fid)

    watertemp = _read_at(fid, frame_position, 332, "<f", layout.frame_header_length, 0.0)
    salinity = _read_at(fid, frame_position, 436, "<I", layout.frame_header_length, 0)
    pressure = _read_at(fid, frame_position, 440, "<f", layout.frame_header_length, 0.0)

    timestamp_str, datenum = _timestamp_from_sonar_microseconds(sonartimestamp)

    return {
        "framenumber": int(framenumber),
        "frametime": int(frametime),
        "version": int(layout.version),
        "flag": int(status),
        "status": int(status),
        "sonartimestamp": int(sonartimestamp),
        "timestamp": timestamp_str,
        "datenum": datenum,
        "year": None,
        "month": None,
        "day": int(tsday),
        "hour": int(tshour),
        "minute": int(tsminute),
        "second": int(tssecond),
        "hsecond": int(tshsecond),
        "transmitmode": int(transmitmode),
        "threshold": int(threshold),
        "intensity": int(intensity),
        "degc1": int(degc1),
        "degc2": int(degc2),
        "humidity": int(humidity),
        "focus": int(focus),
        "battery": int(battery),
        "windowstart": float(windowstart),
        "windowlength": float(windowlength),
        "receivergain": int(receivergain),
        "depth": 0.0,
        "compassheading": 0.0,
        "compasspitch": 0.0,
        "compassroll": 0.0,
        "watertemp": float(watertemp),
        "salinity": int(salinity),
        "pressure": float(pressure),
        "configflags": None,
        "latitude": None,
        "longitude": None,
        "sonarposition": None,
    }


def _parse_frame_header_v01(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    fid.seek(frame_position)
    framenumber = _read_i32(fid)
    frametime = _read_i32(fid)
    _version_ascii = _read_bytes(fid, 4)
    status = _read_i32(fid)
    year = _read_i32(fid)
    month = _read_i32(fid)
    day = _read_i32(fid)
    hour = _read_i32(fid)
    minute = _read_i32(fid)
    second = _read_i32(fid)
    hsecond = _read_i32(fid)
    transmit = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    threshold = _read_i32(fid)
    intensity = _read_i32(fid)
    receivergain = _read_i32(fid)
    degc1 = _read_i32(fid)
    degc2 = _read_i32(fid)
    humidity = _read_i32(fid)
    focus = _read_i32(fid)
    battery = _read_i32(fid)
    status1 = _read_bytes(fid, 16)
    status2 = _read_bytes(fid, 16)

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, layout.resolution, clamp_legacy=True
    )
    configflags = 1
    timestamp_dt = _safe_datetime_utc(year, month, day, hour, minute, second, hsecond)
    sonartimestamp, timestamp_str, datenum = _normalize_datetime(timestamp_dt)

    return {
        "framenumber": int(framenumber),
        "frametime": int(frametime),
        "version": int(layout.version),
        "flag": int(status),
        "status": int(status),
        "sonartimestamp": int(sonartimestamp),
        "timestamp": timestamp_str,
        "datenum": datenum,
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "hour": int(hour),
        "minute": int(minute),
        "second": int(second),
        "hsecond": int(hsecond),
        "transmitmode": int(transmit),
        "threshold": int(threshold),
        "intensity": int(intensity),
        "degc1": int(degc1),
        "degc2": int(degc2),
        "humidity": int(humidity),
        "focus": int(focus),
        "battery": int(battery),
        "windowstart": float(windowstart),
        "windowlength": float(windowlength),
        "receivergain": int(receivergain),
        "depth": 0.0,
        "compassheading": 0.0,
        "compasspitch": 0.0,
        "compassroll": 0.0,
        "watertemp": 0.0,
        "salinity": 0,
        "pressure": 0.0,
        "status1": status1.decode("ascii", errors="ignore").rstrip("\x00"),
        "status2": status2.decode("ascii", errors="ignore").rstrip("\x00"),
        "configflags": int(configflags),
        "latitude": None,
        "longitude": None,
        "sonarposition": None,
    }


def _parse_frame_header_v2(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    fid.seek(frame_position)
    framenumber = _read_i32(fid)
    frametime = _read_i32(fid)
    _version_ascii = _read_bytes(fid, 4)
    status = _read_i32(fid)
    year = _read_i32(fid)
    month = _read_i32(fid)
    day = _read_i32(fid)
    hour = _read_i32(fid)
    minute = _read_i32(fid)
    second = _read_i32(fid)
    hsecond = _read_i32(fid)
    transmit = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    threshold = _read_i32(fid)
    intensity = _read_i32(fid)
    receivergain = _read_i32(fid)
    degc1 = _read_i32(fid)
    degc2 = _read_i32(fid)
    humidity = _read_i32(fid)
    focus = _read_i32(fid)
    battery = _read_i32(fid)
    status1 = _read_bytes(fid, 16)
    status2 = _read_bytes(fid, 16)

    velocity = _read_f32(fid)
    depth = _read_f32(fid)
    altitude = _read_f32(fid)
    pitch = _read_f32(fid)
    pitchrate = _read_f32(fid)
    roll = _read_f32(fid)
    rollrate = _read_f32(fid)
    heading = _read_f32(fid)
    headingrate = _read_f32(fid)
    sonarpan = _read_f32(fid)
    sonartilt = _read_f32(fid)
    latitude = _read_f64(fid)
    longitude = _read_f64(fid)
    fid.read(76)  # userassigned

    windowstart, windowlength, _ = _decode_window_classic(
        windowstart_code, windowlength_code, layout.resolution, clamp_legacy=True
    )
    configflags = 1
    timestamp_dt = _safe_datetime_utc(year, month, day, hour, minute, second, hsecond)
    sonartimestamp, timestamp_str, datenum = _normalize_datetime(timestamp_dt)

    return {
        "framenumber": int(framenumber),
        "frametime": int(frametime),
        "version": int(layout.version),
        "flag": int(status),
        "status": int(status),
        "sonartimestamp": int(sonartimestamp),
        "timestamp": timestamp_str,
        "datenum": datenum,
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "hour": int(hour),
        "minute": int(minute),
        "second": int(second),
        "hsecond": int(hsecond),
        "transmitmode": int(transmit),
        "threshold": int(threshold),
        "intensity": int(intensity),
        "degc1": int(degc1),
        "degc2": int(degc2),
        "humidity": int(humidity),
        "focus": int(focus),
        "battery": int(battery),
        "windowstart": float(windowstart),
        "windowlength": float(windowlength),
        "receivergain": int(receivergain),
        "depth": float(depth),
        "compassheading": float(heading),
        "compasspitch": float(pitch),
        "compassroll": float(roll),
        "watertemp": 0.0,
        "salinity": 0,
        "pressure": 0.0,
        "status1": status1.decode("ascii", errors="ignore").rstrip("\x00"),
        "status2": status2.decode("ascii", errors="ignore").rstrip("\x00"),
        "configflags": int(configflags),
        "velocity": float(velocity),
        "altitude": float(altitude),
        "pitchrate": float(pitchrate),
        "rollrate": float(rollrate),
        "headingrate": float(headingrate),
        "sonarpan": float(sonarpan),
        "sonartilt": float(sonartilt),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "sonarposition": None,
    }


def _parse_frame_header_v3(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    return _parse_frame_header_v34(fid, layout, frame_position, userassigned_size=60)


def _parse_frame_header_v4(fid, layout: DDFLayout, frame_position: int) -> Dict[str, Any]:
    return _parse_frame_header_v34(fid, layout, frame_position, userassigned_size=828)


def _parse_frame_header_v34(
    fid, layout: DDFLayout, frame_position: int, userassigned_size: int
) -> Dict[str, Any]:
    fid.seek(frame_position)
    framenumber = _read_i32(fid)
    frametime = _read_u64(fid)  # MATLAB reads 2x int32; treat as packed 64-bit
    _version_ascii = _read_bytes(fid, 4)
    status = _read_i32(fid)
    year = _read_i32(fid)
    month = _read_i32(fid)
    day = _read_i32(fid)
    hour = _read_i32(fid)
    minute = _read_i32(fid)
    second = _read_i32(fid)
    hsecond = _read_i32(fid)
    transmit = _read_i32(fid)
    windowstart_code = _read_i32(fid)
    windowlength_code = _read_i32(fid)
    threshold = _read_i32(fid)
    intensity = _read_i32(fid)
    receivergain = _read_i32(fid)
    degc1 = _read_i32(fid)
    degc2 = _read_i32(fid)
    humidity = _read_i32(fid)
    focus = _read_i32(fid)
    battery = _read_i32(fid)
    status1 = _read_bytes(fid, 16)
    status2 = _read_bytes(fid, 8)
    panwcom = _read_f32(fid)
    tiltwcom = _read_f32(fid)
    velocity = _read_f32(fid)
    depth = _read_f32(fid)
    altitude = _read_f32(fid)
    pitch = _read_f32(fid)
    pitchrate = _read_f32(fid)
    roll = _read_f32(fid)
    rollrate = _read_f32(fid)
    heading = _read_f32(fid)
    headingrate = _read_f32(fid)
    sonarpan = _read_f32(fid)
    sonartilt = _read_f32(fid)
    sonarroll = _read_f32(fid)
    latitude = _read_f64(fid)
    longitude = _read_f64(fid)
    sonarposition = _read_f32(fid)
    configflags_raw = _read_i32(fid)
    fid.read(userassigned_size)

    configflags = _normalize_configflags(configflags_raw, layout.serialnumber)
    windowstart, windowlength = _decode_window_from_configflags(
        configflags, windowstart_code, windowlength_code, layout.resolution
    )

    timestamp_dt = _safe_datetime_utc(year, month, day, hour, minute, second, hsecond)
    sonartimestamp, timestamp_str, datenum = _normalize_datetime(timestamp_dt)

    return {
        "framenumber": int(framenumber),
        "frametime": int(frametime),
        "version": int(layout.version),
        "flag": int(status),
        "status": int(status),
        "sonartimestamp": int(sonartimestamp),
        "timestamp": timestamp_str,
        "datenum": datenum,
        "year": int(year),
        "month": int(month),
        "day": int(day),
        "hour": int(hour),
        "minute": int(minute),
        "second": int(second),
        "hsecond": int(hsecond),
        "transmitmode": int(transmit),
        "threshold": int(threshold),
        "intensity": int(intensity),
        "degc1": int(degc1),
        "degc2": int(degc2),
        "humidity": int(humidity),
        "focus": int(focus),
        "battery": int(battery),
        "windowstart": float(windowstart),
        "windowlength": float(windowlength),
        "receivergain": int(receivergain),
        "depth": float(depth),
        "compassheading": float(heading),
        "compasspitch": float(pitch),
        "compassroll": float(roll),
        "watertemp": 0.0,
        "salinity": 0,
        "pressure": 0.0,
        "status1": status1.decode("ascii", errors="ignore").rstrip("\x00"),
        "status2": status2.decode("ascii", errors="ignore").rstrip("\x00"),
        "configflags": int(configflags),
        "panwcom": float(panwcom),
        "tiltwcom": float(tiltwcom),
        "velocity": float(velocity),
        "altitude": float(altitude),
        "pitchrate": float(pitchrate),
        "rollrate": float(rollrate),
        "headingrate": float(headingrate),
        "sonarpan": float(sonarpan),
        "sonartilt": float(sonartilt),
        "sonarroll": float(sonarroll),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "sonarposition": float(sonarposition),
    }


def _decode_window_classic(
    windowstart_code: int,
    windowlength_code: int,
    resolution: int,
    clamp_legacy: bool,
):
    index = int(windowlength_code) + 1 + (2 if int(resolution) == 0 else 0)
    if clamp_legacy and index > 5:
        index = 5
    index = max(1, min(index, len(CLASSIC_S_WINDOWS)))

    scale = 0.375 + (0.375 if int(resolution) == 0 else 0.0)
    windowstart = float(windowstart_code) * scale
    windowlength = float(CLASSIC_S_WINDOWS[index - 1])
    return windowstart, windowlength, index


def _decode_window_from_configflags(
    configflags: int, windowstart_code: int, windowlength_code: int, resolution: int
):
    index = int(windowlength_code) + 1 + (2 if int(resolution) == 0 else 0)
    index = max(1, min(index, 6))

    if configflags == 0:
        winlengths = EXTENDED_S_WINDOWS
    elif configflags == 1:
        winlengths = CLASSIC_S_WINDOWS
    elif configflags == 2:
        winlengths = EXTENDED_LR_WINDOWS
    else:
        winlengths = CLASSIC_LR_WINDOWS

    if configflags in (1, 3):
        scale = 0.375 + (0.375 if int(resolution) == 0 else 0.0)
    else:
        scale = 0.419 + (0.419 if int(resolution) == 0 else 0.0)

    windowstart = float(windowstart_code) * scale
    windowlength = float(winlengths[index - 1])
    return windowstart, windowlength


def _configuration_from_flags(configflags: int) -> str:
    mapping = {
        0: "DIDSON-S Extended Windows",
        1: "DIDSON-S Classic Windows",
        2: "DIDSON-LR Extended Windows",
        3: "DIDSON-LR Classic Windows",
    }
    return mapping.get(int(configflags), "Unknown")


def _normalize_configflags(configflags_raw: int, serialnumber: int) -> int:
    configflags = int(configflags_raw) & 3
    if int(serialnumber) < 19:
        configflags = 1
    if int(serialnumber) == 15:
        configflags = 3
    return configflags


def _timestamp_from_sonar_microseconds(sonartimestamp: int):
    if sonartimestamp and sonartimestamp > 0:
        ts_seconds = float(sonartimestamp) * 1e-6
        try:
            dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
            datenum = MATLAB_EPOCH_OFFSET_DAYS + ts_seconds / 86400.0
            timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            return timestamp_str, datenum
        except (OverflowError, OSError, ValueError):
            pass
    return "1970-01-01 00:00:00.000", None


def _safe_datetime_utc(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: int,
    hsecond: int,
) -> Optional[datetime]:
    try:
        millisecond = max(0, min(int(hsecond), 999))
        return datetime(
            int(year),
            int(month),
            int(day),
            int(hour),
            int(minute),
            int(second),
            millisecond * 1000,
            tzinfo=timezone.utc,
        )
    except ValueError:
        return None


def _normalize_datetime(timestamp_dt: Optional[datetime]):
    if timestamp_dt is None:
        return 0, "1970-01-01 00:00:00.000", None
    ts_seconds = timestamp_dt.timestamp()
    sonartimestamp = int(round(ts_seconds * 1e6))
    datenum = MATLAB_EPOCH_OFFSET_DAYS + ts_seconds / 86400.0
    timestamp_str = timestamp_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return sonartimestamp, timestamp_str, datenum


def _read_at(fid, base_offset: int, relative_offset: int, fmt: str, header_len: int, default):
    size = struct.calcsize(fmt)
    if relative_offset < 0 or (relative_offset + size) > int(header_len):
        return default
    current_pos = fid.tell()
    try:
        fid.seek(base_offset + relative_offset)
        return _read_struct(fid, fmt)
    except (EOFError, struct.error):
        return default
    finally:
        fid.seek(current_pos)


def _read_struct(fid, fmt: str):
    size = struct.calcsize(fmt)
    data = fid.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of file while reading format {fmt}")
    values = struct.unpack(fmt, data)
    return values[0] if len(values) == 1 else values


def _read_bytes(fid, size: int) -> bytes:
    data = fid.read(size)
    if len(data) != size:
        raise EOFError(f"Unexpected end of file while reading {size} bytes")
    return data


def _read_i32(fid) -> int:
    return int(_read_struct(fid, "<i"))


def _read_u32(fid) -> int:
    return int(_read_struct(fid, "<I"))


def _read_u64(fid) -> int:
    return int(_read_struct(fid, "<Q"))


def _read_u8(fid) -> int:
    return int(_read_struct(fid, "<B"))


def _read_f32(fid) -> float:
    return float(_read_struct(fid, "<f"))


def _read_f64(fid) -> float:
    return float(_read_struct(fid, "<d"))


FILE_HEADER_PARSERS = {
    0: _parse_file_header_v0,
    1: _parse_file_header_v1,
    2: _parse_file_header_v2,
    3: _parse_file_header_v3,
    4: _parse_file_header_v4,
    5: _parse_file_header_v5,
}

FRAME_HEADER_PARSERS = {
    0: _parse_frame_header_v01,
    1: _parse_frame_header_v01,
    2: _parse_frame_header_v2,
    3: _parse_frame_header_v3,
    4: _parse_frame_header_v4,
    5: _parse_frame_header_v5,
}

