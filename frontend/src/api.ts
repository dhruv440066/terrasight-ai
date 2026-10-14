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

export const API_BASE_URL: string = (() => {
  const envUrl = (import.meta as any).env?.VITE_API_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  if (typeof window !== 'undefined') {
    if (window.location.port === '8000') {
      return window.location.origin;
    }
  }
  return 'http://localhost:8000';
})();

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

apiClient.interceptors.request.use((config) => {
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
  return `${API_BASE_URL}${normalized}`;
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
