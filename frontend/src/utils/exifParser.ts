/**
 * Pure TypeScript EXIF GPS Metadata Parser (Dependency-Free).
 * Reads the APP1 Exif segment from JPEG files via ArrayBuffer to extract
 * GPSLatitude, GPSLatitudeRef, GPSLongitude, and GPSLongitudeRef.
 */

export interface ParsedExifGps {
  latitude: number;
  longitude: number;
  source: 'EXIF_GPS';
  altitude?: number;
  dateTime?: string;
}

export async function extractExifGpsFromBlob(blob: Blob): Promise<ParsedExifGps | null> {
  try {
    // Only need the first 128KB where EXIF header resides
    const headerSlice = blob.slice(0, 131072);
    const arrayBuffer = await headerSlice.arrayBuffer();
    return parseExifFromBuffer(arrayBuffer);
  } catch {
    return null;
  }
}

export function parseExifFromBuffer(buffer: ArrayBuffer): ParsedExifGps | null {
  const data = new DataView(buffer);

  // Check for JPEG SOI marker (0xFFD8)
  if (data.byteLength < 4 || data.getUint16(0, false) !== 0xffd8) {
    return null;
  }

  let offset = 2;
  const length = data.byteLength;

  while (offset + 4 < length) {
    const marker = data.getUint16(offset, false);
    offset += 2;

    if (marker === 0xffe1) {
      // APP1 Marker
      const app1Length = data.getUint16(offset, false);
      offset += 2;

      // Check for 'Exif\0\0' header (0x45786966 0x0000)
      if (
        data.getUint32(offset, false) === 0x45786966 &&
        data.getUint16(offset + 4, false) === 0x0000
      ) {
        const tiffStart = offset + 6;
        return parseTiffGps(data, tiffStart);
      }
      offset += app1Length - 2;
    } else if ((marker & 0xff00) === 0xff00) {
      // Other JPEG marker: skip segment
      if (marker === 0xffda || marker === 0xffd9) {
        // SOS or EOI: headers finished
        break;
      }
      const segmentLength = data.getUint16(offset, false);
      offset += segmentLength;
    } else {
      break;
    }
  }

  return null;
}

function parseTiffGps(data: DataView, tiffStart: number): ParsedExifGps | null {
  if (tiffStart + 8 > data.byteLength) return null;

  // Endianness: 'II' = 0x4949 (Little Endian), 'MM' = 0x4D4D (Big Endian)
  const byteOrder = data.getUint16(tiffStart, false);
  const littleEndian = byteOrder === 0x4949;

  // Validate TIFF marker 42
  if (data.getUint16(tiffStart + 2, littleEndian) !== 0x002a) {
    return null;
  }

  const ifd0Offset = data.getUint32(tiffStart + 4, littleEndian);
  if (tiffStart + ifd0Offset >= data.byteLength) return null;

  // Search IFD0 for GPSInfo IFD pointer (tag 0x8825)
  let gpsOffset = -1;
  const numEntries = data.getUint16(tiffStart + ifd0Offset, littleEndian);
  let entryPtr = tiffStart + ifd0Offset + 2;

  for (let i = 0; i < numEntries; i++) {
    if (entryPtr + 12 > data.byteLength) break;
    const tag = data.getUint16(entryPtr, littleEndian);
    if (tag === 0x8825) {
      gpsOffset = data.getUint32(entryPtr + 8, littleEndian);
      break;
    }
    entryPtr += 12;
  }

  if (gpsOffset === -1 || tiffStart + gpsOffset >= data.byteLength) {
    return null;
  }

  // Parse GPS IFD entries
  const gpsEntriesCount = data.getUint16(tiffStart + gpsOffset, littleEndian);
  let gpsEntryPtr = tiffStart + gpsOffset + 2;

  let latRef = 'N';
  let lonRef = 'E';
  let latDms: [number, number, number] | null = null;
  let lonDms: [number, number, number] | null = null;
  let altitude: number | undefined;

  for (let i = 0; i < gpsEntriesCount; i++) {
    if (gpsEntryPtr + 12 > data.byteLength) break;
    const tag = data.getUint16(gpsEntryPtr, littleEndian);

    if (tag === 0x0001) {
      // GPSLatitudeRef (ASCII 2 bytes)
      latRef = String.fromCharCode(data.getUint8(gpsEntryPtr + 8)).toUpperCase();
    } else if (tag === 0x0002) {
      // GPSLatitude (3 RATIONALs = 24 bytes)
      const valueOffset = data.getUint32(gpsEntryPtr + 8, littleEndian);
      latDms = readDms(data, tiffStart + valueOffset, littleEndian);
    } else if (tag === 0x0003) {
      // GPSLongitudeRef (ASCII 2 bytes)
      lonRef = String.fromCharCode(data.getUint8(gpsEntryPtr + 8)).toUpperCase();
    } else if (tag === 0x0004) {
      // GPSLongitude (3 RATIONALs = 24 bytes)
      const valueOffset = data.getUint32(gpsEntryPtr + 8, littleEndian);
      lonDms = readDms(data, tiffStart + valueOffset, littleEndian);
    } else if (tag === 0x0006) {
      // GPSAltitude (1 RATIONAL = 8 bytes)
      const valueOffset = data.getUint32(gpsEntryPtr + 8, littleEndian);
      const num = data.getUint32(tiffStart + valueOffset, littleEndian);
      const den = data.getUint32(tiffStart + valueOffset + 4, littleEndian);
      if (den > 0) altitude = num / den;
    }

    gpsEntryPtr += 12;
  }

  if (!latDms || !lonDms) {
    return null;
  }

  let lat = latDms[0] + latDms[1] / 60 + latDms[2] / 3600;
  let lon = lonDms[0] + lonDms[1] / 60 + lonDms[2] / 3600;

  if (latRef === 'S') lat = -lat;
  if (lonRef === 'W') lon = -lon;

  // Validate coordinates
  if (lat < -90 || lat > 90 || lon < -180 || lon > 180) {
    return null;
  }
  if (Math.abs(lat) < 1e-6 && Math.abs(lon) < 1e-6) {
    return null;
  }

  return {
    latitude: Math.round(lat * 1e6) / 1e6,
    longitude: Math.round(lon * 1e6) / 1e6,
    source: 'EXIF_GPS',
    altitude: altitude !== undefined ? Math.round(altitude * 10) / 10 : undefined,
  };
}

function readDms(
  data: DataView,
  offset: number,
  littleEndian: boolean
): [number, number, number] | null {
  if (offset + 24 > data.byteLength) return null;

  const degNum = data.getUint32(offset, littleEndian);
  const degDen = data.getUint32(offset + 4, littleEndian);
  const minNum = data.getUint32(offset + 8, littleEndian);
  const minDen = data.getUint32(offset + 12, littleEndian);
  const secNum = data.getUint32(offset + 16, littleEndian);
  const secDen = data.getUint32(offset + 20, littleEndian);

  const deg = degDen > 0 ? degNum / degDen : 0;
  const min = minDen > 0 ? minNum / minDen : 0;
  const sec = secDen > 0 ? secNum / secDen : 0;

  return [deg, min, sec];
}
