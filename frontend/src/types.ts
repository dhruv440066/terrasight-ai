/**
 * Strict TypeScript Interfaces for CleanPath AI (TerraSight AI).
 */

export type UserRole = 'citizen' | 'officer';

export type SeverityTier = 'High' | 'Medium' | 'Low';

export type LocationSource = 'EXIF' | 'MANUAL' | 'CURRENT_LOCATION';

export type IncidentStatus =
  | 'SUBMITTED'
  | 'PENDING'
  | 'VERIFIED'
  | 'ACKNOWLEDGED'
  | 'DISPATCHED'
  | 'IN_PROGRESS'
  | 'RESOLVED'
  | 'REJECTED';

export type HotspotStatus =
  | 'PENDING'
  | 'ACKNOWLEDGED'
  | 'DISPATCHED'
  | 'IN_PROGRESS'
  | 'RESOLVED';

export interface UserProfile {
  id: number;
  name?: string;
  username: string;
  email: string;
  role: UserRole;
  phone?: string;
  created_at: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface WasteDetection {
  id?: number;
  class_name: string;
  category?: string;
  confidence: number;
  bbox: BoundingBox;
  area_ratio?: number;
  object_count?: number;
}

export interface CategoryStat {
  count: number;
  avg_confidence: number;
  max_confidence: number;
}

export interface SeverityBreakdown {
  severity_score: number;
  severity_tier: SeverityTier;
  total_area_ratio?: number;
  effective_multiplier?: number;
  object_count?: number;
  repeat_report_boost?: number;
  formula_explanation?: string;
}

export interface GpsExtractionResult {
  latitude: number | null;
  longitude: number | null;
  source: LocationSource;
}

export interface AnalysisResponse {
  image_path: string;
  image_url: string;
  original_filename?: string;
  image_width: number;
  image_height: number;
  detections: WasteDetection[];
  category_counts: Record<string, number>;
  category_stats?: Record<string, CategoryStat>;
  detected_categories: string[];
  detection_count: number;
  highest_confidence?: number;
  average_confidence?: number;
  severity_score: number;
  severity_tier: SeverityTier;
  severity_breakdown?: SeverityBreakdown;
  inference_mode: 'YOLOv8' | 'Analytical Fallback';
  gps: GpsExtractionResult;
  latitude: number | null;
  longitude: number | null;
  location_source: LocationSource;
  address?: string;
}

export interface AuditLogRecord {
  id: number;
  report_id?: number | null;
  hotspot_id?: number | null;
  actor_id?: number | null;
  actor_name: string;
  actor_role: string;
  action: string;
  old_status?: string | null;
  new_status?: string | null;
  notes: string;
  timestamp: string;
}

export interface DispatchRecord {
  id: number;
  hotspot_id: number;
  hotspot_code?: string;
  officer_id: number;
  officer_username: string;
  crew_name: string;
  vehicle_number?: string | null;
  status: string;
  dispatched_at: string;
  started_at?: string | null;
  completed_at: string | null;
  notes: string;
}

export interface IncidentRecord {
  id: number;
  incident_uuid: string;
  public_report_id?: string;
  citizen_id: number | null;
  citizen_username: string;
  citizen_name?: string;
  citizen_email?: string | null;
  image_path: string;
  image_url: string;
  thumbnail_path?: string;
  original_filename?: string;
  latitude: number;
  longitude: number;
  location_source: LocationSource;
  address?: string;
  description?: string;
  detected_categories: string[];
  category_counts?: Record<string, number>;
  category_stats?: Record<string, CategoryStat>;
  detections: WasteDetection[];
  detection_count: number;
  highest_confidence?: number;
  average_confidence?: number;
  severity_score: number;
  severity_tier: SeverityTier;
  status: IncidentStatus;
  hotspot_id: number | null;
  hotspot_code?: string | null;
  is_primary?: boolean;
  duplicate_status?: 'PRIMARY' | 'DUPLICATE / CLUSTERED';
  distance_from_centroid?: number;
  created_at: string;
  updated_at: string;
  dispatches?: DispatchRecord[];
  audit_logs?: AuditLogRecord[];
}

export interface HotspotRecord {
  id: number;
  hotspot_id: number;
  hotspot_code?: string;
  latitude: number;
  longitude: number;
  address?: string;
  report_count: number;
  duplicate_count?: number;
  duplicate_reports_suppressed: number;
  merge_explanation?: string;
  maximum_severity: number;
  severity_tier: SeverityTier;
  radius_meters: number;
  status: HotspotStatus;
  categories: string[];
  created_at: string;
  updated_at: string;
  incidents: IncidentRecord[];
  dispatches: DispatchRecord[];
  audit_logs?: AuditLogRecord[];
}

export interface SystemStatusInfo {
  ai_engine: string;
  inference_mode: string;
  database: string;
  gis_engine: string;
  clustering_engine: string;
}

export interface AnalyticsSummary {
  total_reports: number;
  reports_today?: number;
  total_hotspots: number;
  active_hotspots: number;
  duplicate_reports_suppressed: number;
  dispatch_operations: number;
  active_dispatches?: number;
  cluster_radius_meters: number;
  high_severity: number;
  medium_severity: number;
  low_severity: number;
  hotspot_severity_breakdown: Record<SeverityTier, number>;
  pending: number;
  acknowledged?: number;
  dispatched: number;
  in_progress: number;
  resolved: number;
  rejected: number;
  category_breakdown: Record<string, number>;
  system_status: SystemStatusInfo;
}
