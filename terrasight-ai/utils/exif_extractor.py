"""
EXIF GPS and Capture Metadata Extractor
=======================================
Extracts geographical coordinates (latitude, longitude, altitude),
capture timestamps, and camera/drone hardware metadata from digital
photographs using Pillow and ExifTags.
"""

from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union
from io import BytesIO
from PIL import Image, ExifTags


def dms_to_decimal(
    dms: Union[Tuple[Any, Any, Any], list],
    ref: Optional[str] = None
) -> Optional[float]:
    """
    Convert GPS Degrees, Minutes, Seconds (DMS) tuple to standard Decimal Degrees (DD).
    
    Args:
        dms: Tuple or list of (degrees, minutes, seconds)
        ref: Direction reference string ('N', 'S', 'E', 'W')
        
    Returns:
        Decimal degrees as float, or None if invalid.
    """
    if not dms or len(dms) < 3:
        return None

    try:
        def _to_float(val: Any) -> float:
            if hasattr(val, "numerator") and hasattr(val, "denominator"):
                return float(val.numerator) / float(val.denominator) if val.denominator != 0 else 0.0
            if isinstance(val, (tuple, list)) and len(val) == 2:
                return float(val[0]) / float(val[1]) if float(val[1]) != 0 else 0.0
            return float(val)

        deg = _to_float(dms[0])
        minutes = _to_float(dms[1])
        seconds = _to_float(dms[2])

        decimal = deg + (minutes / 60.0) + (seconds / 3600.0)

        if ref and str(ref).strip().upper() in ["S", "W"]:
            decimal = -decimal

        return round(decimal, 6)
    except Exception:
        return None


def extract_exif_metadata(
    image_input: Union[str, bytes, BytesIO, Image.Image, Any]
) -> Dict[str, Any]:
    """
    Extract Latitude, Longitude, Altitude, Capture Timestamp, and Device Info.
    
    Args:
        image_input: File path, byte stream, PIL Image, or Streamlit UploadedFile.
        
    Returns:
        Dictionary with structured metadata and validation flags:
        {
            "has_gps": bool,
            "latitude": float or None,
            "longitude": float or None,
            "altitude": float or None,
            "timestamp": str or None,
            "device_make": str or None,
            "device_model": str or None,
            "error": str or None
        }
    """
    result: Dict[str, Any] = {
        "has_gps": False,
        "latitude": None,
        "longitude": None,
        "altitude": None,
        "timestamp": None,
        "device_make": None,
        "device_model": None,
        "error": None,
    }

    try:
        if isinstance(image_input, Image.Image):
            img = image_input
        elif isinstance(image_input, (str, bytes, bytearray, BytesIO)) or hasattr(image_input, "read"):
            if hasattr(image_input, "seek"):
                image_input.seek(0)
            img = Image.open(image_input)
        else:
            result["error"] = "Unsupported image format or object"
            return result

        raw_exif = None
        if hasattr(img, "_getexif"):
            raw_exif = img._getexif()

        if not raw_exif and hasattr(img, "getexif"):
            raw_exif = img.getexif()

        if not raw_exif:
            result["error"] = "No EXIF metadata found in image"
            return result

        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}

        result["device_make"] = tag_map.get("Make")
        result["device_model"] = tag_map.get("Model")

        raw_dt = tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
        if raw_dt:
            try:
                # Format typically 'YYYY:MM:DD HH:MM:SS'
                clean_dt = str(raw_dt).strip()
                dt_obj = datetime.strptime(clean_dt, "%Y:%m:%d %H:%M:%S")
                result["timestamp"] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                result["timestamp"] = str(raw_dt)

        gps_info = tag_map.get("GPSInfo")
        if not gps_info:
            result["error"] = "Image contains EXIF tags, but GPS coordinates are absent"
            return result

        # Map GPS tags using GPSTAGS if dict
        gps_tags = {}
        for k, v in gps_info.items():
            name = ExifTags.GPSTAGS.get(k, k)
            gps_tags[name] = v

        lat_dms = gps_tags.get("GPSLatitude")
        lat_ref = gps_tags.get("GPSLatitudeRef")
        lon_dms = gps_tags.get("GPSLongitude")
        lon_ref = gps_tags.get("GPSLongitudeRef")

        if lat_dms and lon_dms:
            lat = dms_to_decimal(lat_dms, lat_ref)
            lon = dms_to_decimal(lon_dms, lon_ref)

            if lat is not None and lon is not None:
                # Validate range
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    result["has_gps"] = True
                    result["latitude"] = lat
                    result["longitude"] = lon
                else:
                    result["error"] = f"Decoded GPS out of bounds: ({lat}, {lon})"
            else:
                result["error"] = "Failed to convert DMS to decimal coordinates"

        alt_val = gps_tags.get("GPSAltitude")
        alt_ref = gps_tags.get("GPSAltitudeRef")
        if alt_val is not None:
            try:
                if hasattr(alt_val, "numerator") and hasattr(alt_val, "denominator"):
                    alt = float(alt_val.numerator) / float(alt_val.denominator) if alt_val.denominator != 0 else 0.0
                elif isinstance(alt_val, (tuple, list)) and len(alt_val) == 2:
                    alt = float(alt_val[0]) / float(alt_val[1]) if float(alt_val[1]) != 0 else 0.0
                else:
                    alt = float(alt_val)
                if alt_ref and int(alt_ref) == 1:
                    alt = -alt
                result["altitude"] = round(alt, 1)
            except Exception:
                pass

    except Exception as e:
        result["error"] = f"EXIF extraction exception: {str(e)}"

    return result
