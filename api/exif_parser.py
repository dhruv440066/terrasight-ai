"""
EXIF GPS Metadata Extraction Utility for CleanPath AI.
Extracts GPSLatitude, GPSLatitudeRef, GPSLongitude, and GPSLongitudeRef from images
using Pillow and converts Degrees/Minutes/Seconds (DMS) into decimal degrees.
Gracefully falls back to MANUAL location source when EXIF GPS is absent.
"""

import io
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from PIL import ExifTags, Image

logger = logging.getLogger("cleanpath.exif")

GPS_IFD_TAG = 0x8825


def _to_float(value: Any) -> float:
    """Convert EXIF rational, tuple (num, den), int, or float to a Python float."""
    if value is None:
        raise ValueError("Cannot convert None to float")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, tuple) and len(value) == 2:
        num, den = value
        if float(den) == 0.0:
            return 0.0
        return float(num) / float(den)
    if hasattr(value, "numerator") and hasattr(value, "denominator"):
        den = float(value.denominator)
        if den == 0.0:
            return 0.0
        return float(value.numerator) / den
    return float(value)


def dms_to_decimal(
    dms: Union[Tuple[Any, Any, Any], list],
    ref: Optional[Union[str, bytes]] = None,
) -> float:
    """
    Convert Degrees, Minutes, Seconds (DMS) and hemisphere reference (N/S/E/W)
    into signed decimal degrees.

    Formula:
        decimal = degrees + minutes / 60 + seconds / 3600
    """
    if not dms or len(dms) < 3:
        raise ValueError(f"Invalid DMS coordinate sequence: {dms}")

    degrees = _to_float(dms[0])
    minutes = _to_float(dms[1])
    seconds = _to_float(dms[2])

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if isinstance(ref, bytes):
        ref_str = ref.decode("utf-8", errors="ignore").strip().upper()
    elif isinstance(ref, str):
        ref_str = ref.strip().upper()
    else:
        ref_str = ""

    if ref_str in ("S", "W"):
        decimal = -abs(decimal)
    elif ref_str in ("N", "E"):
        decimal = abs(decimal)

    return round(decimal, 6)


def _extract_gps_dict_from_image(img: Image.Image) -> Dict[str, Any]:
    """Extract human-readable GPS tag dictionary from a Pillow Image object."""
    gps_named: Dict[str, Any] = {}

    # Method 1: Modern Pillow getexif() + get_ifd(0x8825)
    try:
        exif_data = img.getexif()
        if exif_data:
            try:
                gps_ifd = exif_data.get_ifd(GPS_IFD_TAG)
                if isinstance(gps_ifd, dict):
                    for key, val in gps_ifd.items():
                        tag_name = ExifTags.GPSTAGS.get(key, str(key))
                        gps_named[tag_name] = val
            except Exception:
                pass

            # Also check if GPSInfo is directly present in exif_data
            if not gps_named and GPS_IFD_TAG in exif_data:
                raw_gps = exif_data[GPS_IFD_TAG]
                if isinstance(raw_gps, dict):
                    for key, val in raw_gps.items():
                        tag_name = ExifTags.GPSTAGS.get(key, str(key))
                        gps_named[tag_name] = val
    except Exception as exc:
        logger.debug("Standard getexif() inspection failed: %s", exc)

    # Method 2: Legacy _getexif() fallback
    if not gps_named and hasattr(img, "_getexif"):
        try:
            raw_exif = img._getexif()  # type: ignore[attr-defined]
            if isinstance(raw_exif, dict):
                for tag_id, value in raw_exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if tag == "GPSInfo" and isinstance(value, dict):
                        for gps_tag_id, gps_val in value.items():
                            sub_tag = ExifTags.GPSTAGS.get(gps_tag_id, str(gps_tag_id))
                            gps_named[sub_tag] = gps_val
        except Exception as exc:
            logger.debug("Legacy _getexif() inspection failed: %s", exc)

    return gps_named


def parse_gps_info(gps_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a dictionary containing GPSLatitude, GPSLatitudeRef, GPSLongitude,
    and GPSLongitudeRef into decimal coordinates.
    """
    manual_fallback: Dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "source": "MANUAL",
    }

    if not gps_info:
        return manual_fallback

    lat_dms = gps_info.get("GPSLatitude")
    lat_ref = gps_info.get("GPSLatitudeRef", "N")
    lon_dms = gps_info.get("GPSLongitude")
    lon_ref = gps_info.get("GPSLongitudeRef", "E")

    if lat_dms is None or lon_dms is None:
        return manual_fallback

    try:
        lat = dms_to_decimal(lat_dms, lat_ref)
        lon = dms_to_decimal(lon_dms, lon_ref)

        # Validate geographic coordinate bounds
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            logger.warning("Extracted GPS out of valid range: (%s, %s)", lat, lon)
            return manual_fallback

        # Reject (0.0, 0.0) null-island corrupted camera tags unless explicitly valid
        if abs(lat) < 1e-7 and abs(lon) < 1e-7:
            return manual_fallback

        return {
            "latitude": lat,
            "longitude": lon,
            "source": "EXIF",
        }
    except Exception as exc:
        logger.warning("Failed to convert EXIF DMS coordinates: %s", exc)
        return manual_fallback


def extract_gps_from_image(
    image_source: Union[str, Path, bytes, io.BytesIO, Image.Image],
) -> Dict[str, Any]:
    """
    Extract GPS coordinates from an image path, raw bytes, BytesIO stream, or PIL Image.
    Never raises an unhandled exception; returns MANUAL source if EXIF is absent or invalid.
    """
    manual_fallback: Dict[str, Any] = {
        "latitude": None,
        "longitude": None,
        "source": "MANUAL",
    }

    try:
        if isinstance(image_source, Image.Image):
            img = image_source
            gps_dict = _extract_gps_dict_from_image(img)
            return parse_gps_info(gps_dict)

        if isinstance(image_source, (bytes, bytearray)):
            stream = io.BytesIO(image_source)
            with Image.open(stream) as img:
                gps_dict = _extract_gps_dict_from_image(img)
                return parse_gps_info(gps_dict)

        if isinstance(image_source, io.BytesIO):
            image_source.seek(0)
            with Image.open(image_source) as img:
                gps_dict = _extract_gps_dict_from_image(img)
                return parse_gps_info(gps_dict)

        path = Path(image_source)
        if not path.exists():
            return manual_fallback

        with Image.open(path) as img:
            gps_dict = _extract_gps_dict_from_image(img)
            return parse_gps_info(gps_dict)

    except Exception as exc:
        logger.warning("EXIF extraction gracefully handled error: %s", exc)
        return manual_fallback
