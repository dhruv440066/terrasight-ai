import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import { Camera, Compass, MapPin, Navigation } from 'lucide-react';
import { reverseGeocodeApi } from '../api';
import type { LocationSource } from '../types';

// Configure default Leaflet marker icon so it renders reliably in Vite builds
const pinIcon = L.divIcon({
  className: 'custom-hotspot-marker',
  html: `<div style="
    width: 28px;
    height: 28px;
    background: #DC2626;
    border: 3px solid #FFFFFF;
    border-radius: 50% 50% 50% 0;
    transform: rotate(-45deg);
    box-shadow: 0 3px 8px rgba(0,0,0,0.35);
  "></div>`,
  iconSize: [28, 28],
  iconAnchor: [14, 28],
});

interface MapPickerProps {
  latitude: number | null;
  longitude: number | null;
  locationSource: LocationSource;
  exifCoords?: { latitude: number | null; longitude: number | null };
  address?: string;
  onChangeCoordinates: (
    lat: number,
    lon: number,
    source: LocationSource,
    resolvedAddress?: string
  ) => void;
}

const DEFAULT_CENTER: [number, number] = [28.6139, 77.209];

const MapClickAndRecenter: React.FC<{
  lat: number;
  lon: number;
  onPick: (lat: number, lon: number) => void;
}> = ({ lat, lon, onPick }) => {
  const map = useMap();

  useEffect(() => {
    map.setView([lat, lon], map.getZoom(), { animate: true });
  }, [lat, lon, map]);

  useMapEvents({
    click(e) {
      onPick(
        Number(e.latlng.lat.toFixed(6)),
        Number(e.latlng.lng.toFixed(6))
      );
    },
  });

  return null;
};

export const MapPicker: React.FC<MapPickerProps> = ({
  latitude,
  longitude,
  locationSource,
  exifCoords,
  address,
  onChangeCoordinates,
}) => {
  const activeLat = latitude !== null ? latitude : DEFAULT_CENTER[0];
  const activeLon = longitude !== null ? longitude : DEFAULT_CENTER[1];
  const markerRef = useRef<L.Marker | null>(null);
  const [resolvedAddress, setResolvedAddress] = useState<string>(
    address || 'Connaught Place, New Delhi, India'
  );
  const [isGeocoding, setIsGeocoding] = useState<boolean>(false);

  const hasExifGps =
    exifCoords != null &&
    exifCoords.latitude !== null &&
    exifCoords.longitude !== null;

  useEffect(() => {
    let cancelled = false;
    setIsGeocoding(true);
    reverseGeocodeApi(activeLat, activeLon)
      .then((res) => {
        if (!cancelled && res.address) {
          setResolvedAddress(res.address);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setResolvedAddress(
            `Municipal Coordinate (${activeLat.toFixed(4)}°N, ${activeLon.toFixed(4)}°E), New Delhi, India`
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsGeocoding(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [activeLat, activeLon]);

  const eventHandlers = useMemo(
    () => ({
      dragend() {
        const marker = markerRef.current;
        if (marker != null) {
          const pos = marker.getLatLng();
          onChangeCoordinates(
            Number(pos.lat.toFixed(6)),
            Number(pos.lng.toFixed(6)),
            'MANUAL'
          );
        }
      },
    }),
    [onChangeCoordinates]
  );

  const handleUseBrowserGeolocation = () => {
    if (!navigator.geolocation) {
      onChangeCoordinates(DEFAULT_CENTER[0], DEFAULT_CENTER[1], 'CURRENT_LOCATION');
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        onChangeCoordinates(
          Number(pos.coords.latitude.toFixed(6)),
          Number(pos.coords.longitude.toFixed(6)),
          'CURRENT_LOCATION'
        );
      },
      () => {
        onChangeCoordinates(DEFAULT_CENTER[0], DEFAULT_CENTER[1], 'CURRENT_LOCATION');
      },
      { enableHighAccuracy: true, timeout: 6000 }
    );
  };

  const handleSelectExifGps = () => {
    if (hasExifGps && exifCoords?.latitude != null && exifCoords?.longitude != null) {
      onChangeCoordinates(exifCoords.latitude, exifCoords.longitude, 'EXIF');
    }
  };

  const sourceLabel =
    locationSource === 'EXIF'
      ? 'EXIF GPS'
      : locationSource === 'CURRENT_LOCATION'
      ? 'Current Location'
      : 'Manual Pin';

  return (
    <div className="space-y-3.5 rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <MapPin className="h-5 w-5 text-emerald-600" />
          <span className="text-sm font-bold text-slate-900">
            Location Confirmation & Interactive Pin Picker
          </span>
        </div>
        <button
          type="button"
          onClick={handleUseBrowserGeolocation}
          className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-100"
        >
          <Navigation className="h-3.5 w-3.5 text-emerald-600" />
          Use Current Location / Default Pin
        </button>
      </div>

      {/* 3-Way Location Source Selector */}
      <div className="grid grid-cols-3 gap-2 rounded-lg bg-slate-100 p-1 text-xs">
        <button
          type="button"
          disabled={!hasExifGps}
          onClick={handleSelectExifGps}
          className={`flex items-center justify-center gap-1.5 rounded-md px-2.5 py-2 font-semibold transition ${
            locationSource === 'EXIF'
              ? 'bg-white text-emerald-700 shadow-sm ring-1 ring-emerald-500/30'
              : hasExifGps
              ? 'text-slate-700 hover:bg-white/60'
              : 'cursor-not-allowed text-slate-400'
          }`}
          title={
            hasExifGps
              ? 'Use GPS coordinates embedded in photo EXIF'
              : 'No EXIF GPS metadata found in uploaded image'
          }
        >
          <Camera className="h-3.5 w-3.5" />
          <span>EXIF GPS</span>
        </button>

        <button
          type="button"
          onClick={handleUseBrowserGeolocation}
          className={`flex items-center justify-center gap-1.5 rounded-md px-2.5 py-2 font-semibold transition ${
            locationSource === 'CURRENT_LOCATION'
              ? 'bg-white text-emerald-700 shadow-sm ring-1 ring-emerald-500/30'
              : 'text-slate-700 hover:bg-white/60'
          }`}
        >
          <Compass className="h-3.5 w-3.5" />
          <span>Current Location</span>
        </button>

        <button
          type="button"
          onClick={() => onChangeCoordinates(activeLat, activeLon, 'MANUAL')}
          className={`flex items-center justify-center gap-1.5 rounded-md px-2.5 py-2 font-semibold transition ${
            locationSource === 'MANUAL'
              ? 'bg-white text-emerald-700 shadow-sm ring-1 ring-emerald-500/30'
              : 'text-slate-700 hover:bg-white/60'
          }`}
        >
          <MapPin className="h-3.5 w-3.5" />
          <span>Manual Pin</span>
        </button>
      </div>

      <p className="text-xs text-slate-600">
        Click anywhere on the map or drag the red marker to pinpoint the exact waste location.
      </p>

      <div className="h-64 w-full overflow-hidden rounded-lg border border-slate-200">
        <MapContainer
          center={[activeLat, activeLon]}
          zoom={15}
          scrollWheelZoom={true}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <MapClickAndRecenter
            lat={activeLat}
            lon={activeLon}
            onPick={(lat, lon) => onChangeCoordinates(lat, lon, 'MANUAL')}
          />
          <Marker
            draggable={true}
            eventHandlers={eventHandlers}
            position={[activeLat, activeLon]}
            icon={pinIcon}
            ref={markerRef}
          />
        </MapContainer>
      </div>

      {/* Reverse Geocoded Address Display */}
      <div className="rounded-lg border border-emerald-200/80 bg-emerald-50/50 px-3.5 py-2.5">
        <span className="block text-[10px] font-bold uppercase tracking-wider text-emerald-700">
          Reverse-Geocoded Municipal Address
        </span>
        <p className="mt-0.5 text-xs font-semibold text-slate-800">
          {isGeocoding ? 'Resolving street & ward address...' : resolvedAddress}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 pt-0.5 sm:grid-cols-3">
        <div className="rounded-lg bg-slate-50 px-3 py-2">
          <span className="block text-[11px] font-semibold uppercase text-slate-500">
            Latitude
          </span>
          <input
            type="number"
            step="0.0001"
            value={latitude !== null ? latitude : ''}
            placeholder="e.g. 28.6139"
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!Number.isNaN(val)) {
                onChangeCoordinates(val, activeLon, 'MANUAL');
              }
            }}
            className="mt-0.5 w-full bg-transparent text-sm font-bold text-slate-900 focus:outline-none"
          />
        </div>
        <div className="rounded-lg bg-slate-50 px-3 py-2">
          <span className="block text-[11px] font-semibold uppercase text-slate-500">
            Longitude
          </span>
          <input
            type="number"
            step="0.0001"
            value={longitude !== null ? longitude : ''}
            placeholder="e.g. 77.2090"
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              if (!Number.isNaN(val)) {
                onChangeCoordinates(activeLat, val, 'MANUAL');
              }
            }}
            className="mt-0.5 w-full bg-transparent text-sm font-bold text-slate-900 focus:outline-none"
          />
        </div>
        <div className="rounded-lg bg-slate-50 px-3 py-2">
          <span className="block text-[11px] font-semibold uppercase text-slate-500">
            Location Source
          </span>
          <span className="mt-0.5 block text-sm font-bold text-emerald-700">
            {sourceLabel}
          </span>
        </div>
      </div>
    </div>
  );
};
