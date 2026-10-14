/**
 * Geospatial Calculation Utilities for CleanPath AI.
 * Implements Haversine great-circle distance in meters, radian conversion,
 * 50m radius proximity checks, and coordinate formatting.
 */

export const EARTH_RADIUS_METERS = 6371000;

/**
 * Convert degrees to radians.
 */
export function toRadians(degrees: number): number {
  return (degrees * Math.PI) / 180;
}

/**
 * Calculate the great-circle distance in meters between two geographic coordinates
 * using the Haversine formula.
 */
export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number
): number {
  const phi1 = toRadians(lat1);
  const phi2 = toRadians(lat2);
  const deltaPhi = toRadians(lat2 - lat1);
  const deltaLambda = toRadians(lon2 - lon1);

  const a =
    Math.sin(deltaPhi / 2) * Math.sin(deltaPhi / 2) +
    Math.cos(phi1) *
      Math.cos(phi2) *
      Math.sin(deltaLambda / 2) *
      Math.sin(deltaLambda / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(Math.max(0, 1 - a)));
  return EARTH_RADIUS_METERS * c;
}

/**
 * Determine whether two geographic points lie within a specified radius in meters.
 * Default radius is 50 meters (matching the DBSCAN hotspot epsilon).
 */
export function isWithinRadius(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
  radiusMeters: number = 50
): boolean {
  return haversineDistance(lat1, lon1, lat2, lon2) <= radiusMeters;
}

/**
 * Format latitude and longitude to a readable fixed-precision string.
 */
export function formatCoordinates(
  latitude: number | null | undefined,
  longitude: number | null | undefined,
  decimals: number = 5
): string {
  if (
    latitude === null ||
    latitude === undefined ||
    longitude === null ||
    longitude === undefined ||
    Number.isNaN(latitude) ||
    Number.isNaN(longitude)
  ) {
    return 'Coordinates unavailable';
  }
  return `${latitude.toFixed(decimals)}, ${longitude.toFixed(decimals)}`;
}
