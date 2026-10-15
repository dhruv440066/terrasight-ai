/**
 * Axios API Client for CleanPath AI.
 * Connects React frontend (http://localhost:3000) to FastAPI backend (http://localhost:8000).
 */

import axios from 'axios';
import type {
  AnalyticsSummary,
  AnalysisResponse,
  AuthResponse,
  DispatchRecord,
  HotspotRecord,
  IncidentRecord,
  IncidentStatus,
  HotspotStatus,
  LocationSource,
  SeverityTier,
  UserProfile,
  WasteDetection,
} from './types';

export function normalizeApiUrl(rawUrl: string): string {
  let url = (rawUrl || '').trim().replace(/\/+$/, '');
  if (!url) return '';
  if (!/^https?:\/\//i.test(url)) {
    if (/^(localhost|127\.0\.0\.1)(:\d+)?/i.test(url)) {
      url = `http://${url}`;
    } else {
      url = `https://${url}`;
    }
  }
  return url;
}

export function determineApiBaseUrl(): string {
  // 1. Explicit user override stored in localStorage
  if (typeof window !== 'undefined') {
    const saved =
      localStorage.getItem('terrasight_api_url') ||
      localStorage.getItem('cleanpath_api_url');
    if (saved && saved.trim().length > 0) {
      return normalizeApiUrl(saved);
    }
  }

  // 2. Vite build-time environment variable (VITE_API_URL)
  const envUrl = (import.meta as any).env?.VITE_API_URL;
  if (
    envUrl &&
    typeof envUrl === 'string' &&
    envUrl.trim().length > 0 &&
    !envUrl.includes('%VITE_API_URL%')
  ) {
    return normalizeApiUrl(envUrl);
  }

  // 3. Smart browser runtime detection
  if (typeof window !== 'undefined') {
    const { hostname, origin, port } = window.location;

    // A. Localhost development environments
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      if (port === '8000') {
        return origin; // Backend serving frontend directly on port 8000
      }
      return 'http://localhost:8000'; // Vite dev server proxying to local backend
    }

    // B. Render Cloud deployments (*.onrender.com)
    if (hostname.endsWith('.onrender.com')) {
      // If hosted directly on the backend web service (e.g. terrasight-api.onrender.com)
      if (hostname.includes('api') && !hostname.includes('frontend')) {
        return origin;
      }
      // If hosted on a static frontend, derive the backend API URL
      if (hostname.includes('frontend')) {
        return `https://${hostname.replace('frontend', 'api')}`;
      }
      if (hostname.includes('ui')) {
        return `https://${hostname.replace('ui', 'api')}`;
      }
      // Standard Render blueprint fallback
      return 'https://terrasight-api.onrender.com';
    }

    // C. Custom domains or other hosting platforms (Netlify, Vercel, VPS)
    if (port === '' || port === '80' || port === '443') {
      // If deployed unified on the backend, origin is the API
      // Otherwise default to the standard production backend on Render
      return 'https://terrasight-api.onrender.com';
    }
  }

  return 'http://localhost:8000';
}

export let API_BASE_URL: string = determineApiBaseUrl();

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export function setApiBaseUrl(newUrl: string): void {
  const normalized = normalizeApiUrl(newUrl);
  API_BASE_URL = normalized;
  apiClient.defaults.baseURL = normalized;
  if (typeof window !== 'undefined') {
    localStorage.setItem('terrasight_api_url', normalized);
    window.dispatchEvent(
      new CustomEvent('terrasight_api_url_changed', { detail: normalized })
    );
  }
}

export function resetApiBaseUrl(): void {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('terrasight_api_url');
    localStorage.removeItem('cleanpath_api_url');
  }
  const detected = determineApiBaseUrl();
  API_BASE_URL = detected;
  apiClient.defaults.baseURL = detected;
  if (typeof window !== 'undefined') {
    window.dispatchEvent(
      new CustomEvent('terrasight_api_url_changed', { detail: detected })
    );
  }
}

export async function testBackendHealth(
  targetUrl?: string
): Promise<{ ok: boolean; message: string; latencyMs: number; raw?: any }> {
  const target = normalizeApiUrl(targetUrl || getApiBaseUrl());
  const start = Date.now();
  try {
    const res = await axios.get(`${target}/health`, { timeout: 30000 });
    const latency = Date.now() - start;
    if (res.status === 200 && (res.data?.status === 'ok' || res.data?.service)) {
      return {
        ok: true,
        message: `Connected (${latency}ms) — All Subsystems Online`,
        latencyMs: latency,
        raw: res.data,
      };
    }
    return {
      ok: false,
      message: `Server responded with HTTP ${res.status}`,
      latencyMs: latency,
      raw: res.data,
    };
  } catch (err: any) {
    const latency = Date.now() - start;
    if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
      return {
        ok: false,
        message:
          'Connection timed out. On Render free tier, backend takes 30-50s to wake up from sleep.',
        latencyMs: latency,
      };
    }
    const errMsg =
      err.response?.data?.detail || err.message || 'Unable to connect to backend server';
    return {
      ok: false,
      message: `Failed to connect (${errMsg})`,
      latencyMs: latency,
    };
  }
}

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60s timeout accommodates Render free-tier cold starts
});

apiClient.interceptors.request.use((config) => {
  // Ensure baseURL is always synchronized with the current active URL
  config.baseURL = getApiBaseUrl();

  const token =
    localStorage.getItem('terrasight_token') ||
    localStorage.getItem('cleanpath_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export function resolveImageUrl(pathOrUrl: string): string {
  if (!pathOrUrl) return '';
  if (
    pathOrUrl.startsWith('http://') ||
    pathOrUrl.startsWith('https://') ||
    pathOrUrl.startsWith('blob:') ||
    pathOrUrl.startsWith('data:')
  ) {
    return pathOrUrl;
  }
  const normalized = pathOrUrl.startsWith('/') ? pathOrUrl : `/uploads/${pathOrUrl}`;
  return `${getApiBaseUrl()}${normalized}`;
}

export async function loginApi(username: string, password: string): Promise<AuthResponse> {
  const response = await apiClient.post<AuthResponse>('/auth/login', {
    username,
    password,
  });
  return response.data;
}

export async function registerApi(
  username: string,
  email: string,
  password: string,
  role: 'citizen' | 'officer' = 'citizen'
): Promise<AuthResponse> {
  const response = await apiClient.post<AuthResponse>('/auth/register', {
    username,
    email,
    password,
    role,
  });
  return response.data;
}

export async function fetchCurrentUser(): Promise<UserProfile> {
  const response = await apiClient.get<UserProfile>('/auth/me');
  return response.data;
}

export async function analyzeWasteImageApi(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const response = await apiClient.post<AnalysisResponse>('/incidents/analyze', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return response.data;
}

export async function reverseGeocodeApi(
  latitude: number,
  longitude: number
): Promise<{ address: string; ward: string; city: string; source: string }> {
  const response = await apiClient.get('/reverse-geocode', {
    params: { latitude, longitude },
  });
  return response.data;
}

export interface CreateIncidentPayload {
  image_path: string;
  original_filename?: string;
  latitude: number;
  longitude: number;
  location_source: LocationSource;
  address?: string;
  description?: string;
  detected_categories: string[];
  detections: WasteDetection[];
  detection_count: number;
  severity_score: number;
  severity_tier: SeverityTier;
}

export async function submitIncidentApi(payload: CreateIncidentPayload): Promise<IncidentRecord> {
  const response = await apiClient.post<IncidentRecord>('/incidents', payload);
  return response.data;
}

export async function fetchMyIncidentsApi(guestUuids?: string[]): Promise<IncidentRecord[]> {
  const params: Record<string, string> = {};
  if (guestUuids && guestUuids.length > 0) {
    params.guest_uuids = guestUuids.join(',');
  }
  const response = await apiClient.get<IncidentRecord[]>('/incidents/my', { params });
  return response.data;
}

export async function fetchIncidentByIdApi(incidentId: string | number): Promise<IncidentRecord> {
  const response = await apiClient.get<IncidentRecord>(
    `/incidents/${encodeURIComponent(String(incidentId))}`
  );
  return response.data;
}

export async function fetchOfficerIncidentsApi(): Promise<IncidentRecord[]> {
  const response = await apiClient.get<IncidentRecord[]>('/officer/incidents');
  return response.data;
}

export async function fetchOfficerHotspotsApi(): Promise<HotspotRecord[]> {
  const response = await apiClient.get<HotspotRecord[]>('/officer/hotspots');
  return response.data;
}

export async function fetchOfficerAnalyticsApi(): Promise<AnalyticsSummary> {
  const response = await apiClient.get<AnalyticsSummary>('/officer/analytics');
  return response.data;
}

export async function dispatchSanitationCrewApi(
  hotspotId: number,
  crewName: string,
  notes: string,
  vehicleNumber?: string
): Promise<{ message: string; dispatch: DispatchRecord; hotspot: HotspotRecord }> {
  const response = await apiClient.post<{
    message: string;
    dispatch: DispatchRecord;
    hotspot: HotspotRecord;
  }>(`/officer/dispatch/${hotspotId}`, {
    crew_name: crewName,
    vehicle_number: vehicleNumber || undefined,
    notes,
  });
  return response.data;
}

export async function updateIncidentStatusApi(
  incidentId: number | string,
  status: IncidentStatus,
  notes?: string
): Promise<IncidentRecord> {
  const response = await apiClient.patch<IncidentRecord>(
    `/officer/incidents/${incidentId}/status`,
    { status, notes }
  );
  return response.data;
}

export async function updateHotspotStatusApi(
  hotspotId: number,
  status: HotspotStatus,
  notes?: string
): Promise<HotspotRecord> {
  const response = await apiClient.patch<HotspotRecord>(
    `/officer/hotspots/${hotspotId}/status`,
    { status, notes }
  );
  return response.data;
}

export async function seedDemoDataApi(
  centerLatitude: number = 28.6139,
  centerLongitude: number = 77.209
): Promise<{
  message: string;
  total_reports: number;
  unique_hotspots: number;
  duplicate_reports_suppressed: number;
  dispatch_operations: number;
}> {
  const response = await apiClient.post('/demo/seed', {
    center_latitude: centerLatitude,
    center_longitude: centerLongitude,
    clear_existing: true,
  });
  return response.data;
}

export async function clearDemoDataApi(): Promise<{ removed_incidents: number }> {
  const response = await apiClient.post('/demo/clear');
  return response.data;
}
