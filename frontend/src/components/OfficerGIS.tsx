import React, { useCallback, useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Circle as LeafletCircle,
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  useMap,
} from 'react-leaflet';
import L from 'leaflet';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  CheckCircle2,
  Database,
  Eye,
  FileText,
  Layers,
  Leaf,
  LogOut,
  Map as MapIcon,
  MapPin,
  RefreshCw,
  Shield,
  Sparkles,
  Trash2,
  Truck,
} from 'lucide-react';
import {
  clearDemoDataApi,
  dispatchSanitationCrewApi,
  fetchOfficerAnalyticsApi,
  fetchOfficerHotspotsApi,
  fetchOfficerIncidentsApi,
  resolveImageUrl,
  seedDemoDataApi,
  updateHotspotStatusApi,
  updateIncidentStatusApi,
} from '../api';
import type {
  AnalyticsSummary,
  HotspotRecord,
  HotspotStatus,
  IncidentRecord,
  IncidentStatus,
  SeverityTier,
  UserProfile,
} from '../types';
import { formatCoordinates } from '../utils/geoCalculations';
import { DispatchModal } from './DispatchModal';
import { ReportDetailModal } from './ReportDetailModal';

interface OfficerGISProps {
  user: UserProfile;
  onLogout: () => void;
}

const SEVERITY_COLORS: Record<
  SeverityTier,
  { hex: string; badge: string; dot: string; ring: string }
> = {
  High: {
    hex: '#DC2626',
    badge: 'bg-red-600 text-white',
    dot: '🔴',
    ring: 'border-red-500 bg-red-50 text-red-900',
  },
  Medium: {
    hex: '#EAB308',
    badge: 'bg-amber-500 text-slate-950',
    dot: '🟡',
    ring: 'border-amber-500 bg-amber-50 text-amber-900',
  },
  Low: {
    hex: '#16A34A',
    badge: 'bg-emerald-600 text-white',
    dot: '🟢',
    ring: 'border-emerald-500 bg-emerald-50 text-emerald-900',
  },
};

function createHotspotDivIcon(
  tier: SeverityTier,
  reportCount: number,
  isSelected: boolean
): L.DivIcon {
  const color = SEVERITY_COLORS[tier]?.hex || '#16A34A';
  const size = isSelected ? 38 : 32;
  const border = isSelected ? '4px solid #0F172A' : '3px solid #FFFFFF';

  return L.divIcon({
    className: 'custom-hotspot-marker',
    html: `<div style="
      width: ${size}px;
      height: ${size}px;
      background: ${color};
      border: ${border};
      border-radius: 9999px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #FFFFFF;
      font-weight: 900;
      font-size: 12px;
      box-shadow: 0 4px 12px rgba(15, 23, 42, 0.45);
    ">${reportCount}</div>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
    popupAnchor: [0, -size / 2],
  });
}

function createReportPinDivIcon(
  tier: SeverityTier,
  isPrimary: boolean
): L.DivIcon {
  const color = SEVERITY_COLORS[tier]?.hex || '#16A34A';
  const border = isPrimary ? '2px solid #0F172A' : '2px dashed #FFFFFF';

  return L.divIcon({
    className: 'custom-report-pin-marker',
    html: `<div style="
      width: 16px;
      height: 16px;
      background: ${color};
      border: ${border};
      border-radius: 4px;
      transform: rotate(45deg);
      box-shadow: 0 2px 6px rgba(15, 23, 42, 0.35);
    "></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    popupAnchor: [0, -8],
  });
}

const MapFocusController: React.FC<{
  target: [number, number] | null;
}> = ({ target }) => {
  const map = useMap();
  useEffect(() => {
    if (target) {
      map.flyTo(target, Math.max(map.getZoom(), 16), { duration: 0.6 });
    }
  }, [target, map]);
  return null;
};

export const OfficerGIS: React.FC<OfficerGISProps> = ({ user, onLogout }) => {
  const location = useLocation();
  const navigate = useNavigate();

  const initialView = location.pathname.includes('/analytics')
    ? 'analytics'
    : 'command';
  const [activeView, setActiveView] = useState<'command' | 'analytics'>(
    initialView
  );

  useEffect(() => {
    if (location.pathname.includes('/analytics')) {
      setActiveView('analytics');
    } else {
      setActiveView('command');
    }
  }, [location.pathname]);

  const switchView = (view: 'command' | 'analytics') => {
    setActiveView(view);
    if (view === 'analytics') navigate('/officer/analytics');
    else navigate('/officer/map');
  };

  const [hotspots, setHotspots] = useState<HotspotRecord[]>([]);
  const [incidents, setIncidents] = useState<IncidentRecord[]>([]);
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [selectedHotspotId, setSelectedHotspotId] = useState<number | null>(
    null
  );
  const [mapFocusTarget, setMapFocusTarget] = useState<[number, number] | null>(
    null
  );
  const [showIndividualPins, setShowIndividualPins] = useState<boolean>(true);
  const [showHotspotReports, setShowHotspotReports] = useState<boolean>(true);
  const [dispatchModalHotspot, setDispatchModalHotspot] =
    useState<HotspotRecord | null>(null);
  const [selectedReportModal, setSelectedReportModal] =
    useState<IncidentRecord | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [seeding, setSeeding] = useState<boolean>(false);
  const [bannerMessage, setBannerMessage] = useState<string | null>(null);

  const loadCommandData = useCallback(async () => {
    setLoading(true);
    try {
      const [hotspotsData, analyticsData, incidentsData] = await Promise.all([
        fetchOfficerHotspotsApi(),
        fetchOfficerAnalyticsApi(),
        fetchOfficerIncidentsApi(),
      ]);
      setHotspots(hotspotsData);
      setAnalytics(analyticsData);
      setIncidents(incidentsData);

      setSelectedHotspotId((prevId) => {
        if (prevId && hotspotsData.some((h) => h.id === prevId)) {
          return prevId;
        }
        return hotspotsData.length > 0 ? hotspotsData[0].id : null;
      });
    } catch {
      // Handled gracefully
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadCommandData();
  }, [loadCommandData]);

  const selectedHotspot =
    hotspots.find((h) => h.id === selectedHotspotId) || hotspots[0] || null;

  const handleSeedDemo = async () => {
    setSeeding(true);
    setBannerMessage(null);
    try {
      const res = await seedDemoDataApi();
      setBannerMessage(
        `Demo Mode Activated: ${res.total_reports} citizen reports clustered into ${res.unique_hotspots} unique hotspots (${res.duplicate_reports_suppressed} duplicate reports suppressed) with ${res.dispatch_operations} dispatch operations.`
      );
      await loadCommandData();
    } finally {
      setSeeding(false);
    }
  };

  const handleClearDemo = async () => {
    setSeeding(true);
    setBannerMessage(null);
    try {
      const res = await clearDemoDataApi();
      setBannerMessage(
        `Cleared ${res.removed_incidents} synthetic demo incidents. Displaying only real citizen reports.`
      );
      await loadCommandData();
    } finally {
      setSeeding(false);
    }
  };

  const handleConfirmDispatch = async (
    hotspotId: number,
    crewName: string,
    notes: string
  ) => {
    const res = await dispatchSanitationCrewApi(hotspotId, crewName, notes);
    setBannerMessage(res.message);
    await loadCommandData();
  };

  const handleHotspotStatusChange = async (
    hotspotId: number,
    newStatus: HotspotStatus
  ) => {
    await updateHotspotStatusApi(hotspotId, newStatus);
    setBannerMessage(`Hotspot #${hotspotId} status updated to ${newStatus}.`);
    await loadCommandData();
  };

  const handleIncidentStatusChange = async (
    incidentId: number | string,
    newStatus: IncidentStatus
  ) => {
    const updated = await updateIncidentStatusApi(incidentId, newStatus);
    setBannerMessage(
      `Report ${updated.incident_uuid} status updated to ${newStatus}.`
    );
    if (selectedReportModal && selectedReportModal.id === updated.id) {
      setSelectedReportModal(updated);
    }
    await loadCommandData();
  };

  const handleViewReportOnMap = (
    lat: number,
    lon: number,
    hotspotId?: number | null
  ) => {
    setActiveView('command');
    if (hotspotId) {
      setSelectedHotspotId(hotspotId);
    }
    setMapFocusTarget([lat, lon]);
  };

  const mapCenter: [number, number] = mapFocusTarget
    ? mapFocusTarget
    : selectedHotspot
    ? [selectedHotspot.latitude, selectedHotspot.longitude]
    : [28.6139, 77.209];

  const sysStatus = analytics?.system_status || {
    ai_engine: 'Online',
    inference_mode: 'YOLOv8',
    database: 'Online',
    gis_engine: 'Online',
    clustering_engine: 'Online',
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFC] text-[#0F172A]">
      {/* Command Center Top Navigation */}
      <header className="sticky top-0 z-30 bg-[#0F172A] text-white shadow-md">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-600">
              <Leaf className="h-5 w-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="text-base font-black tracking-tight">
                  CLEANPATH AI
                </span>
                <span className="rounded bg-emerald-500/20 px-2 py-0.5 text-[10px] font-bold uppercase text-emerald-400">
                  Municipal Command Center
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                50m Haversine DBSCAN Hotspot Deduplication &amp; Crew Dispatch
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => switchView('command')}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                activeView === 'command'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <MapIcon className="h-3.5 w-3.5" />
              GIS Command Map
            </button>

            <button
              type="button"
              onClick={() => switchView('analytics')}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-bold transition ${
                activeView === 'analytics'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              <BarChart3 className="h-3.5 w-3.5" />
              Analytics &amp; Registry
            </button>

            <button
              type="button"
              disabled={seeding}
              onClick={handleSeedDemo}
              className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-500/40 bg-emerald-950/60 px-3 py-1.5 text-xs font-bold text-emerald-300 hover:bg-emerald-900/70 disabled:opacity-50"
            >
              <Sparkles className="h-3.5 w-3.5" />
              {seeding ? 'Working...' : 'Seed Demo Data (15→10)'}
            </button>

            <button
              type="button"
              disabled={seeding}
              onClick={handleClearDemo}
              title="Remove synthetic demo incidents and display only real citizen submissions"
              className="inline-flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-800/80 px-2.5 py-1.5 text-xs font-semibold text-slate-300 hover:bg-slate-700 hover:text-white disabled:opacity-50"
            >
              <Trash2 className="h-3.5 w-3.5 text-slate-400" />
              <span className="hidden lg:inline">Clear Demo</span>
            </button>

            <div className="hidden items-center gap-1.5 rounded-lg bg-slate-800 px-3 py-1.5 text-xs font-semibold text-slate-200 md:flex">
              <Shield className="h-3.5 w-3.5 text-emerald-400" />
              <span>Officer: {user.name || user.username}</span>
            </div>

            <button
              type="button"
              onClick={onLogout}
              className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-200 hover:border-red-600 hover:bg-red-600 hover:text-white"
            >
              <LogOut className="h-3.5 w-3.5" />
              Logout
            </button>
          </div>
        </div>

        {/* System Status Sub-Bar */}
        <div className="border-t border-slate-800 bg-slate-900/90 px-4 py-1.5 text-[11px] text-slate-300 sm:px-6">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap items-center gap-4">
              <span className="inline-flex items-center gap-1.5">
                <span
                  className={`h-2 w-2 rounded-full ${
                    sysStatus.ai_engine === 'Online'
                      ? 'bg-emerald-400'
                      : 'bg-amber-400'
                  }`}
                />
                <strong>AI Engine:</strong> {sysStatus.ai_engine} (
                {sysStatus.inference_mode})
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <strong>Database:</strong> {sysStatus.database}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <strong>GIS Engine:</strong> {sysStatus.gis_engine}
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-emerald-400" />
                <strong>Clustering Engine:</strong> {sysStatus.clustering_engine} (50m
                Haversine DBSCAN)
              </span>
            </div>

            <button
              type="button"
              onClick={loadCommandData}
              className="inline-flex items-center gap-1 font-semibold text-emerald-400 hover:underline"
            >
              <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
              Sync Live GIS
            </button>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl flex-1 space-y-5 px-4 py-5 sm:px-6">
        {bannerMessage && (
          <div className="flex items-center justify-between rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-2.5 text-xs font-bold text-emerald-900">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-600" />
              <span>{bannerMessage}</span>
            </div>
            <button
              type="button"
              onClick={() => setBannerMessage(null)}
              className="text-emerald-700 hover:underline"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* DBSCAN Deduplication Demonstration Pipeline Strip + Live DB KPIs */}
        {analytics && (
          <div className="space-y-3">
            <div className="grid grid-cols-1 gap-3 rounded-2xl border border-slate-200 bg-white p-4 shadow-xs sm:grid-cols-4">
              <div className="flex items-center justify-between rounded-xl bg-slate-50 px-4 py-3">
                <div>
                  <span className="text-[11px] font-bold uppercase text-slate-500">
                    Citizen Reports
                  </span>
                  <p className="text-2xl font-black text-slate-900">
                    {analytics.total_reports}
                  </p>
                  <span className="text-[11px] text-slate-500">
                    {analytics.reports_today !== undefined
                      ? `${analytics.reports_today} submitted today`
                      : 'Individual photo submissions'}
                  </span>
                </div>
                <ArrowRight className="hidden h-5 w-5 text-slate-400 sm:block" />
              </div>

              <div className="flex items-center justify-between rounded-xl bg-emerald-50/70 px-4 py-3">
                <div>
                  <span className="text-[11px] font-bold uppercase text-emerald-800">
                    Unique Hotspots (50m DBSCAN)
                  </span>
                  <p className="text-2xl font-black text-emerald-700">
                    {analytics.total_hotspots}
                  </p>
                  <span className="text-[11px] font-semibold text-emerald-800">
                    50-meter Haversine clusters
                  </span>
                </div>
                <ArrowRight className="hidden h-5 w-5 text-emerald-500 sm:block" />
              </div>

              <div className="flex items-center justify-between rounded-xl bg-amber-50/80 px-4 py-3">
                <div>
                  <span className="text-[11px] font-bold uppercase text-amber-900">
                    Duplicate Reports Suppressed
                  </span>
                  <p className="text-2xl font-black text-amber-700">
                    {analytics.duplicate_reports_suppressed}
                  </p>
                  <span className="text-[11px] font-semibold text-amber-800">
                    Dispatch deduplicated, 0 deleted
                  </span>
                </div>
                <ArrowRight className="hidden h-5 w-5 text-amber-500 sm:block" />
              </div>

              <div className="flex items-center justify-between rounded-xl bg-slate-900 px-4 py-3 text-white">
                <div>
                  <span className="text-[11px] font-bold uppercase text-emerald-400">
                    Active Dispatches
                  </span>
                  <p className="text-2xl font-black text-white">
                    {analytics.dispatch_operations}
                  </p>
                  <span className="text-[11px] text-slate-300">
                    Sanitation crews deployed
                  </span>
                </div>
                <Truck className="h-6 w-6 text-emerald-400" />
              </div>
            </div>

            {/* Secondary Live Operational Statistics Bar (Section 14 & 36) */}
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  Reports Today
                </span>
                <p className="mt-0.5 text-lg font-black text-slate-900">
                  {analytics.reports_today ?? analytics.total_reports}
                </p>
              </div>
              <div className="rounded-xl border border-red-200 bg-red-50/40 px-4 py-2.5 shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-red-700">
                  High Severity Reports
                </span>
                <p className="mt-0.5 text-lg font-black text-red-600">
                  {analytics.high_severity}
                </p>
              </div>
              <div className="rounded-xl border border-amber-200 bg-amber-50/40 px-4 py-2.5 shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-amber-800">
                  Pending / Acknowledged
                </span>
                <p className="mt-0.5 text-lg font-black text-amber-700">
                  {analytics.pending}
                </p>
              </div>
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 px-4 py-2.5 shadow-2xs">
                <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-800">
                  Resolved Reports
                </span>
                <p className="mt-0.5 text-lg font-black text-emerald-700">
                  {analytics.resolved}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* VIEW 1: GIS COMMAND MAP + PRIORITY QUEUE + HOTSPOT DETAILS */}
        {activeView === 'command' && (
          <div className="space-y-5">
            {/* GIS Map Card */}
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 px-5 py-3">
                <div className="flex items-center gap-2">
                  <MapIcon className="h-5 w-5 text-emerald-600" />
                  <h2 className="text-sm font-extrabold uppercase tracking-wide text-slate-900">
                    Municipal GIS Command Map (50m Cluster Radius)
                  </h2>
                </div>
                <div className="flex flex-wrap items-center gap-3 text-xs font-bold">
                  <button
                    type="button"
                    onClick={() => setShowIndividualPins((prev) => !prev)}
                    className={`inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] font-bold transition ${
                      showIndividualPins
                        ? 'border-emerald-600 bg-emerald-50 text-emerald-800'
                        : 'border-slate-200 bg-slate-50 text-slate-500'
                    }`}
                  >
                    <MapPin className="h-3 w-3" />
                    {showIndividualPins
                      ? `Individual Report Pins (${incidents.length})`
                      : 'Show Report Pins'}
                  </button>
                  <span className="inline-flex items-center gap-1.5 text-red-600">
                    <span>🔴 High (&gt;60%)</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5 text-amber-600">
                    <span>🟡 Medium (25–60%)</span>
                  </span>
                  <span className="inline-flex items-center gap-1.5 text-emerald-600">
                    <span>🟢 Low (&lt;25%)</span>
                  </span>
                  <span className="rounded bg-slate-100 px-2 py-0.5 text-[11px] text-slate-700">
                    --- 50m Dashed Radius
                  </span>
                </div>
              </div>

              <div className="h-[430px] w-full">
                <MapContainer
                  center={mapCenter}
                  zoom={15}
                  scrollWheelZoom={true}
                  className="h-full w-full"
                >
                  <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                  />
                  <MapFocusController target={mapCenter} />

                  {/* Layer 1: 50m Dashed Hotspot Circles & Centroid Markers */}
                  {hotspots.map((hotspot) => {
                    const colorHex =
                      SEVERITY_COLORS[hotspot.severity_tier]?.hex || '#16A34A';
                    const isSelected = selectedHotspot?.id === hotspot.id;

                    return (
                      <React.Fragment key={hotspot.id}>
                        {/* 50-meter dashed cluster radius circle */}
                        <LeafletCircle
                          center={[hotspot.latitude, hotspot.longitude]}
                          radius={hotspot.radius_meters || 50}
                          pathOptions={{
                            color: colorHex,
                            weight: isSelected ? 3 : 2,
                            dashArray: '6, 6',
                            fillColor: colorHex,
                            fillOpacity: isSelected ? 0.22 : 0.12,
                          }}
                        />

                        {/* Hotspot Centroid Marker */}
                        <Marker
                          position={[hotspot.latitude, hotspot.longitude]}
                          icon={createHotspotDivIcon(
                            hotspot.severity_tier,
                            hotspot.report_count,
                            isSelected
                          )}
                          eventHandlers={{
                            click: () => {
                              setSelectedHotspotId(hotspot.id);
                              setMapFocusTarget([
                                hotspot.latitude,
                                hotspot.longitude,
                              ]);
                            },
                          }}
                        >
                          <Popup>
                            <div className="min-w-[225px] space-y-2 p-1 text-xs">
                              <div className="flex items-center justify-between border-b border-slate-200 pb-1.5">
                                <span className="text-sm font-black text-slate-900">
                                  {hotspot.hotspot_code || `Hotspot #${hotspot.id}`}
                                </span>
                                <span
                                  className={`rounded px-2 py-0.5 text-[10px] font-extrabold uppercase ${
                                    SEVERITY_COLORS[hotspot.severity_tier].badge
                                  }`}
                                >
                                  {hotspot.severity_tier.toUpperCase()}
                                </span>
                              </div>

                              <div className="space-y-1 text-slate-700">
                                <div>
                                  <strong>Max Severity:</strong>{' '}
                                  {hotspot.maximum_severity} / 100
                                </div>
                                <div>
                                  <strong>Clustered Reports:</strong>{' '}
                                  {hotspot.report_count} (
                                  {hotspot.duplicate_reports_suppressed} duplicates
                                  merged)
                                </div>
                                <div>
                                  <strong>Address:</strong>{' '}
                                  {hotspot.address ||
                                    formatCoordinates(
                                      hotspot.latitude,
                                      hotspot.longitude,
                                      5
                                    )}
                                </div>
                                <div>
                                  <strong>Categories:</strong>
                                  <div className="mt-0.5 flex flex-wrap gap-1">
                                    {hotspot.categories.map((cat) => (
                                      <span
                                        key={cat}
                                        className="rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-semibold"
                                      >
                                        {cat}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                                <div>
                                  <strong>Status:</strong>{' '}
                                  <span className="font-bold text-slate-900">
                                    {hotspot.status}
                                  </span>
                                </div>
                              </div>

                              <button
                                type="button"
                                onClick={() => setDispatchModalHotspot(hotspot)}
                                className="mt-2 w-full rounded-lg bg-emerald-600 px-3 py-1.5 text-center text-xs font-bold text-white hover:bg-emerald-700"
                              >
                                Dispatch Sanitation Crew
                              </button>
                            </div>
                          </Popup>
                        </Marker>
                      </React.Fragment>
                    );
                  })}

                  {/* Layer 2: Individual Citizen Report Markers (Section 21 & 39) */}
                  {showIndividualPins &&
                    incidents.map((inc) => (
                      <Marker
                        key={`inc-pin-${inc.id}`}
                        position={[inc.latitude, inc.longitude]}
                        icon={createReportPinDivIcon(
                          inc.severity_tier,
                          inc.is_primary !== false
                        )}
                      >
                        <Popup>
                          <div className="min-w-[215px] space-y-2 p-1 text-xs">
                            <div className="flex items-center justify-between border-b border-slate-200 pb-1">
                              <span className="font-black text-slate-900">
                                {inc.incident_uuid}
                              </span>
                              <span
                                className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                                  SEVERITY_COLORS[inc.severity_tier].badge
                                }`}
                              >
                                {inc.severity_tier} ({inc.severity_score})
                              </span>
                            </div>

                            <img
                              src={resolveImageUrl(inc.image_url || inc.image_path)}
                              alt={inc.incident_uuid}
                              className="h-24 w-full rounded object-cover"
                            />

                            <div className="space-y-0.5 text-[11px] text-slate-700">
                              <div>
                                <strong>Category:</strong>{' '}
                                {inc.detected_categories.join(', ')}
                              </div>
                              <div>
                                <strong>Status:</strong> {inc.status}
                              </div>
                              <div>
                                <strong>Cluster Role:</strong>{' '}
                                {inc.duplicate_status || 'PRIMARY'}
                              </div>
                            </div>

                            <button
                              type="button"
                              onClick={() => setSelectedReportModal(inc)}
                              className="w-full rounded-lg bg-slate-900 px-3 py-1.5 text-center text-xs font-bold text-white hover:bg-slate-800"
                            >
                              VIEW REPORT
                            </button>
                          </div>
                        </Popup>
                      </Marker>
                    ))}
                </MapContainer>
              </div>
            </div>

            {/* Bottom Split Layout: Priority Queue + Hotspot Details */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-12">
              {/* Left Column: Severity Priority Queue */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-5">
                <div className="mb-4 flex items-center justify-between border-b border-slate-100 pb-3">
                  <div>
                    <h3 className="text-sm font-extrabold uppercase tracking-wider text-slate-900">
                      Priority Queue
                    </h3>
                    <p className="text-[11px] text-slate-500">
                      Sorted by Severity Tier (High → Medium → Low) &amp; Max Score
                    </p>
                  </div>
                  <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-bold text-slate-700">
                    {hotspots.length} Hotspots
                  </span>
                </div>

                {hotspots.length === 0 ? (
                  <div className="rounded-xl border border-dashed border-slate-200 bg-slate-50 p-8 text-center">
                    <MapPin className="mx-auto mb-2 h-8 w-8 text-slate-400" />
                    <p className="text-sm font-bold text-slate-700">
                      No active waste hotspots.
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Submit a citizen waste report or click &ldquo;Seed Demo
                      Data&rdquo; above to populate sample clusters.
                    </p>
                  </div>
                ) : (
                  <div className="max-h-[520px] space-y-2.5 overflow-y-auto pr-1">
                    {hotspots.map((hotspot) => {
                      const isSelected = selectedHotspot?.id === hotspot.id;
                      const style = SEVERITY_COLORS[hotspot.severity_tier];

                      return (
                        <button
                          key={hotspot.id}
                          type="button"
                          onClick={() => {
                            setSelectedHotspotId(hotspot.id);
                            setMapFocusTarget([
                              hotspot.latitude,
                              hotspot.longitude,
                            ]);
                          }}
                          className={`w-full rounded-xl border p-3.5 text-left transition ${
                            isSelected
                              ? 'border-slate-900 bg-slate-900 text-white shadow-sm'
                              : 'border-slate-200 bg-slate-50/70 hover:bg-slate-100'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span
                                className={`rounded px-2 py-0.5 text-[11px] font-black uppercase ${style.badge}`}
                              >
                                {hotspot.severity_tier.toUpperCase()}{' '}
                                {hotspot.hotspot_code || `#${hotspot.id}`}
                              </span>
                              <span
                                className={`text-sm font-black ${
                                  isSelected ? 'text-white' : 'text-slate-900'
                                }`}
                              >
                                {hotspot.maximum_severity} / 100
                              </span>
                            </div>
                            <span
                              className={`rounded-md px-2 py-0.5 text-[11px] font-bold ${
                                isSelected
                                  ? 'bg-slate-800 text-emerald-400'
                                  : 'bg-white text-slate-700'
                              }`}
                            >
                              {hotspot.report_count}{' '}
                              {hotspot.report_count === 1 ? 'Report' : 'Reports'}
                            </span>
                          </div>

                          <div className="mt-1.5 truncate text-[11px] font-medium opacity-80">
                            {hotspot.address ||
                              formatCoordinates(
                                hotspot.latitude,
                                hotspot.longitude,
                                4
                              )}
                          </div>

                          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                            <span
                              className={
                                isSelected ? 'text-slate-300' : 'text-slate-500'
                              }
                            >
                              {hotspot.categories.join(' • ')}
                            </span>
                            <span
                              className={`font-bold uppercase ${
                                hotspot.status === 'RESOLVED'
                                  ? 'text-emerald-400'
                                  : hotspot.status === 'DISPATCHED'
                                  ? 'text-amber-400'
                                  : isSelected
                                  ? 'text-slate-300'
                                  : 'text-slate-600'
                              }`}
                            >
                              {hotspot.status}
                            </span>
                          </div>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Right Column: Hotspot Details & Constituent Reports Auditability */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-7">
                {selectedHotspot ? (
                  <div className="space-y-5">
                    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 pb-4">
                      <div>
                        <div className="flex items-center gap-2.5">
                          <h3 className="text-lg font-black text-slate-900">
                            {selectedHotspot.hotspot_code ||
                              `Hotspot #${selectedHotspot.id}`}{' '}
                            Details
                          </h3>
                          <span
                            className={`rounded-md px-2.5 py-0.5 text-xs font-extrabold uppercase ${
                              SEVERITY_COLORS[selectedHotspot.severity_tier].badge
                            }`}
                          >
                            {selectedHotspot.severity_tier.toUpperCase()}
                          </span>
                        </div>
                        <p className="mt-1 text-xs font-semibold text-slate-700">
                          {selectedHotspot.address}
                        </p>
                        <p className="mt-0.5 text-[11px] text-slate-500">
                          Centroid:{' '}
                          <strong>
                            {formatCoordinates(
                              selectedHotspot.latitude,
                              selectedHotspot.longitude,
                              5
                            )}
                          </strong>{' '}
                          • Cluster Radius:{' '}
                          <strong>{selectedHotspot.radius_meters}m</strong>
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center gap-2">
                        <select
                          aria-label="Hotspot Status"
                          value={selectedHotspot.status}
                          onChange={(e) =>
                            handleHotspotStatusChange(
                              selectedHotspot.id,
                              e.target.value as HotspotStatus
                            )
                          }
                          className="rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs font-bold text-slate-800 focus:border-emerald-600 focus:outline-none"
                        >
                          <option value="PENDING">Status: PENDING</option>
                          <option value="ACKNOWLEDGED">Status: ACKNOWLEDGED</option>
                          <option value="DISPATCHED">Status: DISPATCHED</option>
                          <option value="IN_PROGRESS">Status: IN_PROGRESS</option>
                          <option value="RESOLVED">Status: RESOLVED</option>
                        </select>

                        <button
                          type="button"
                          onClick={() => setDispatchModalHotspot(selectedHotspot)}
                          className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-extrabold text-white shadow-xs hover:bg-emerald-700"
                        >
                          <Truck className="h-4 w-4" />
                          DISPATCH CREW
                        </button>
                      </div>
                    </div>

                    {/* Duplicate Suppression Explanation Banner (Section 40) */}
                    <div className="rounded-xl border border-amber-200 bg-amber-50/70 px-4 py-2.5 text-xs font-semibold text-amber-900">
                      {selectedHotspot.merge_explanation ||
                        `${selectedHotspot.report_count} citizen reports within ${selectedHotspot.radius_meters}m merged into 1 dispatch hotspot`}
                    </div>

                    {/* Key Metrics Grid */}
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                      <div className="rounded-xl bg-slate-50 p-3">
                        <span className="text-[11px] font-semibold text-slate-500">
                          Reports Clustered
                        </span>
                        <p className="mt-0.5 text-xl font-black text-slate-900">
                          {selectedHotspot.report_count}
                        </p>
                      </div>
                      <div className="rounded-xl bg-amber-50/70 p-3">
                        <span className="text-[11px] font-semibold text-amber-800">
                          Duplicates Suppressed
                        </span>
                        <p className="mt-0.5 text-xl font-black text-amber-700">
                          {selectedHotspot.duplicate_reports_suppressed}
                        </p>
                      </div>
                      <div className="rounded-xl bg-slate-50 p-3">
                        <span className="text-[11px] font-semibold text-slate-500">
                          Maximum Severity
                        </span>
                        <p className="mt-0.5 text-xl font-black text-slate-900">
                          {selectedHotspot.maximum_severity} / 100
                        </p>
                      </div>
                      <div className="rounded-xl bg-slate-50 p-3">
                        <span className="text-[11px] font-semibold text-slate-500">
                          Current Status
                        </span>
                        <p className="mt-0.5 text-sm font-black text-emerald-700">
                          {selectedHotspot.status}
                        </p>
                      </div>
                    </div>

                    {/* Categories */}
                    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-slate-100 bg-slate-50/60 px-4 py-3 text-xs">
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span className="font-bold text-slate-700">
                          Detected Categories:
                        </span>
                        {selectedHotspot.categories.map((cat) => (
                          <span
                            key={cat}
                            className="rounded-md bg-white px-2.5 py-1 font-bold text-slate-800 shadow-2xs"
                          >
                            {cat}
                          </span>
                        ))}
                      </div>
                    </div>

                    {/* Active Dispatches */}
                    {selectedHotspot.dispatches.length > 0 ? (
                      <div className="space-y-2 rounded-xl border border-emerald-200 bg-emerald-50/60 p-3.5 text-xs">
                        <div className="font-extrabold uppercase tracking-wider text-emerald-900">
                          Sanitation Crew Dispatches (
                          {selectedHotspot.dispatches.length})
                        </div>
                        {selectedHotspot.dispatches.map((disp) => (
                          <div
                            key={disp.id}
                            className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-white p-2.5 shadow-2xs"
                          >
                            <div>
                              <span className="font-bold text-slate-900">
                                {disp.crew_name}
                              </span>
                              {disp.notes && (
                                <p className="text-[11px] text-slate-600">
                                  {disp.notes}
                                </p>
                              )}
                            </div>
                            <span className="rounded bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                              {disp.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs text-slate-500">
                        No sanitation dispatches. Click{' '}
                        <strong>DISPATCH CREW</strong> to assign a team.
                      </div>
                    )}

                    {/* Auditability: View Underlying Citizen Reports (Section 40 & 49) */}
                    <div className="border-t border-slate-100 pt-4">
                      <div className="mb-3 flex items-center justify-between">
                        <button
                          type="button"
                          onClick={() =>
                            setShowHotspotReports(!showHotspotReports)
                          }
                          className="inline-flex items-center gap-1.5 text-xs font-extrabold uppercase tracking-wide text-emerald-700 hover:underline"
                        >
                          <Eye className="h-4 w-4" />
                          {showHotspotReports
                            ? `Hide Underlying Citizen Reports (${selectedHotspot.incidents.length})`
                            : `View Reports (${selectedHotspot.incidents.length})`}
                        </button>
                        <span className="text-[11px] text-slate-500">
                          Audit trail preserved (zero data deletion)
                        </span>
                      </div>

                      {showHotspotReports && (
                        <div className="max-h-72 space-y-2.5 overflow-y-auto pr-1">
                          {selectedHotspot.incidents.map((inc, idx) => (
                            <div
                              key={inc.id}
                              className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50/60 p-3 text-xs"
                            >
                              <div className="flex items-center gap-3">
                                <img
                                  src={resolveImageUrl(
                                    inc.image_url || inc.image_path
                                  )}
                                  alt={inc.incident_uuid}
                                  className="h-12 w-16 rounded-lg object-cover"
                                />
                                <div>
                                  <div className="flex flex-wrap items-center gap-1.5">
                                    <span className="font-extrabold text-slate-900">
                                      Report #{idx + 1} — {inc.incident_uuid}
                                    </span>
                                    <span
                                      className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                                        inc.is_primary !== false
                                          ? 'bg-emerald-100 text-emerald-800'
                                          : 'bg-amber-100 text-amber-800'
                                      }`}
                                    >
                                      {inc.duplicate_status ||
                                        (idx === 0
                                          ? 'PRIMARY'
                                          : 'DUPLICATE / CLUSTERED')}
                                    </span>
                                  </div>
                                  <div className="text-[11px] text-slate-600">
                                    Severity:{' '}
                                    <strong>{inc.severity_score} / 100</strong> (
                                    {inc.severity_tier}) • {inc.location_source}
                                    {inc.distance_from_centroid !== undefined &&
                                      ` • ${inc.distance_from_centroid}m from centroid`}
                                  </div>
                                  <div className="text-[11px] text-slate-500">
                                    Categories: {inc.detected_categories.join(', ')}
                                  </div>
                                </div>
                              </div>

                              <div className="flex items-center gap-2">
                                <button
                                  type="button"
                                  onClick={() => setSelectedReportModal(inc)}
                                  className="inline-flex items-center gap-1 rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-[11px] font-bold text-slate-700 hover:bg-slate-100"
                                >
                                  <Eye className="h-3.5 w-3.5 text-emerald-600" />
                                  VIEW REPORT
                                </button>

                                <select
                                  aria-label={`Status for ${inc.incident_uuid}`}
                                  value={inc.status}
                                  onChange={(e) =>
                                    handleIncidentStatusChange(
                                      inc.id,
                                      e.target.value as IncidentStatus
                                    )
                                  }
                                  className="rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-[11px] font-bold text-slate-800"
                                >
                                  <option value="SUBMITTED">SUBMITTED</option>
                                  <option value="ACKNOWLEDGED">ACKNOWLEDGED</option>
                                  <option value="VERIFIED">VERIFIED</option>
                                  <option value="DISPATCHED">DISPATCHED</option>
                                  <option value="IN_PROGRESS">IN_PROGRESS</option>
                                  <option value="RESOLVED">RESOLVED</option>
                                  <option value="REJECTED">REJECTED</option>
                                </select>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="p-10 text-center text-sm text-slate-500">
                    <FileText className="mx-auto mb-2 h-9 w-9 text-slate-400" />
                    <p className="font-bold text-slate-700">
                      No citizen reports yet.
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      When citizens submit waste photographs, 50m DBSCAN hotspots and
                      priority dispatch controls will appear here automatically.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 2: ANALYTICS DASHBOARD & COMPLETE CITIZEN REPORT REGISTRY */}
        {activeView === 'analytics' && analytics && (
          <div className="space-y-6">
            {/* Summary Metric Cards */}
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-slate-500">
                  Total Reports
                </span>
                <p className="mt-1 text-2xl font-black text-slate-900">
                  {analytics.total_reports}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-emerald-700">
                  Active Hotspots
                </span>
                <p className="mt-1 text-2xl font-black text-emerald-600">
                  {analytics.active_hotspots}
                </p>
              </div>
              <div className="rounded-xl border border-red-200 bg-red-50/40 p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-red-700">
                  High Severity
                </span>
                <p className="mt-1 text-2xl font-black text-red-600">
                  {analytics.high_severity}
                </p>
              </div>
              <div className="rounded-xl border border-amber-200 bg-amber-50/40 p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-amber-800">
                  Medium Severity
                </span>
                <p className="mt-1 text-2xl font-black text-amber-600">
                  {analytics.medium_severity}
                </p>
              </div>
              <div className="rounded-xl border border-emerald-200 bg-emerald-50/40 p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-emerald-800">
                  Low Severity
                </span>
                <p className="mt-1 text-2xl font-black text-emerald-600">
                  {analytics.low_severity}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-slate-500">
                  Dispatched
                </span>
                <p className="mt-1 text-2xl font-black text-indigo-600">
                  {analytics.dispatched}
                </p>
              </div>
              <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-2xs">
                <span className="text-[11px] font-bold uppercase text-slate-500">
                  Resolved
                </span>
                <p className="mt-1 text-2xl font-black text-emerald-700">
                  {analytics.resolved}
                </p>
              </div>
            </div>

            {/* Visual Distribution Charts */}
            <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
              {/* Chart 1: Severity Distribution */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-extrabold uppercase tracking-wide text-slate-900">
                  <AlertTriangle className="h-4 w-4 text-red-600" />
                  Severity Distribution
                </h3>
                <div className="space-y-3 text-xs">
                  {[
                    {
                      label: 'High (>60%)',
                      val: analytics.high_severity,
                      color: 'bg-red-600',
                    },
                    {
                      label: 'Medium (25–60%)',
                      val: analytics.medium_severity,
                      color: 'bg-amber-500',
                    },
                    {
                      label: 'Low (<25%)',
                      val: analytics.low_severity,
                      color: 'bg-emerald-600',
                    },
                  ].map((item) => {
                    const pct =
                      analytics.total_reports > 0
                        ? Math.round((item.val / analytics.total_reports) * 100)
                        : 0;
                    return (
                      <div key={item.label} className="space-y-1">
                        <div className="flex justify-between font-bold">
                          <span>{item.label}</span>
                          <span>
                            {item.val} ({pct}%)
                          </span>
                        </div>
                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                          <div
                            className={`h-full ${item.color}`}
                            style={{ width: `${Math.max(4, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Chart 2: Reports by Functional Waste Category */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-extrabold uppercase tracking-wide text-slate-900">
                  <Layers className="h-4 w-4 text-emerald-600" />
                  Detected Objects by Category
                </h3>
                <div className="space-y-3 text-xs">
                  {Object.entries(analytics.category_breakdown).map(
                    ([cat, count]) => {
                      const totalCatCount = Math.max(
                        1,
                        Object.values(analytics.category_breakdown).reduce(
                          (a, b) => a + b,
                          0
                        )
                      );
                      const pct = Math.min(
                        100,
                        Math.round((count / totalCatCount) * 100)
                      );
                      return (
                        <div key={cat} className="space-y-1">
                          <div className="flex justify-between font-bold">
                            <span>{cat}</span>
                            <span>{count} objects</span>
                          </div>
                          <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                            <div
                              className="h-full bg-[#0F172A]"
                              style={{ width: `${Math.max(4, pct)}%` }}
                            />
                          </div>
                        </div>
                      );
                    }
                  )}
                </div>
              </div>

              {/* Chart 3: Incident Status Breakdown */}
              <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-xs">
                <h3 className="mb-4 flex items-center gap-2 text-sm font-extrabold uppercase tracking-wide text-slate-900">
                  <Activity className="h-4 w-4 text-indigo-600" />
                  Incident Status
                </h3>
                <div className="space-y-3 text-xs">
                  {[
                    {
                      label: 'Pending / Acknowledged',
                      val: analytics.pending,
                      color: 'bg-amber-500',
                    },
                    {
                      label: 'Dispatched / In Progress',
                      val: analytics.dispatched,
                      color: 'bg-indigo-600',
                    },
                    {
                      label: 'Resolved',
                      val: analytics.resolved,
                      color: 'bg-emerald-600',
                    },
                  ].map((item) => {
                    const pct =
                      analytics.total_reports > 0
                        ? Math.round((item.val / analytics.total_reports) * 100)
                        : 0;
                    return (
                      <div key={item.label} className="space-y-1">
                        <div className="flex justify-between font-bold">
                          <span>{item.label}</span>
                          <span>
                            {item.val} ({pct}%)
                          </span>
                        </div>
                        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
                          <div
                            className={`h-full ${item.color}`}
                            style={{ width: `${Math.max(4, pct)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Complete Citizen Incident Registry */}
            <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xs">
              <div className="flex items-center justify-between border-b border-slate-100 px-5 py-4">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-emerald-600" />
                  <h3 className="text-sm font-extrabold uppercase tracking-wide text-slate-900">
                    Incoming Citizen Reports Registry ({incidents.length})
                  </h3>
                </div>
              </div>
              {incidents.length === 0 ? (
                <div className="p-8 text-center text-sm text-slate-500">
                  No citizen reports yet.
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-[11px] font-bold uppercase text-slate-500">
                      <tr>
                        <th className="px-4 py-3">Report ID</th>
                        <th className="px-4 py-3">Hotspot</th>
                        <th className="px-4 py-3">Categories</th>
                        <th className="px-4 py-3">Severity</th>
                        <th className="px-4 py-3">Location / Address</th>
                        <th className="px-4 py-3">Status</th>
                        <th className="px-4 py-3">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {incidents.map((inc) => (
                        <tr key={inc.id} className="hover:bg-slate-50/80">
                          <td className="px-4 py-3 font-bold text-slate-900">
                            {inc.incident_uuid}
                          </td>
                          <td className="px-4 py-3 font-semibold text-emerald-700">
                            {inc.hotspot_code || `Hotspot #${inc.hotspot_id}`}
                          </td>
                          <td className="px-4 py-3">
                            {inc.detected_categories.join(', ')}
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`rounded px-2 py-0.5 text-[10px] font-extrabold uppercase ${
                                SEVERITY_COLORS[inc.severity_tier].badge
                              }`}
                            >
                              {inc.severity_tier} ({inc.severity_score})
                            </span>
                          </td>
                          <td className="px-4 py-3 text-[11px]">
                            <div className="font-semibold text-slate-800">
                              {inc.address ||
                                formatCoordinates(inc.latitude, inc.longitude, 4)}
                            </div>
                            <div className="font-mono text-[10px] text-slate-500">
                              {formatCoordinates(inc.latitude, inc.longitude, 4)} (
                              {inc.location_source})
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <select
                              aria-label={`Table status for ${inc.incident_uuid}`}
                              value={inc.status}
                              onChange={(e) =>
                                handleIncidentStatusChange(
                                  inc.id,
                                  e.target.value as IncidentStatus
                                )
                              }
                              className="rounded border border-slate-300 bg-white px-2 py-1 text-[11px] font-bold"
                            >
                              <option value="SUBMITTED">SUBMITTED</option>
                              <option value="ACKNOWLEDGED">ACKNOWLEDGED</option>
                              <option value="VERIFIED">VERIFIED</option>
                              <option value="DISPATCHED">DISPATCHED</option>
                              <option value="IN_PROGRESS">IN_PROGRESS</option>
                              <option value="RESOLVED">RESOLVED</option>
                              <option value="REJECTED">REJECTED</option>
                            </select>
                          </td>
                          <td className="px-4 py-3">
                            <button
                              type="button"
                              onClick={() => setSelectedReportModal(inc)}
                              className="inline-flex items-center gap-1 rounded-lg bg-slate-900 px-2.5 py-1 text-[11px] font-bold text-white hover:bg-slate-800"
                            >
                              <Eye className="h-3 w-3" />
                              VIEW REPORT
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Dispatch Modal */}
      <DispatchModal
        hotspot={dispatchModalHotspot}
        isOpen={dispatchModalHotspot !== null}
        onClose={() => setDispatchModalHotspot(null)}
        onConfirmDispatch={handleConfirmDispatch}
      />

      {/* Full Report Detail Modal */}
      <ReportDetailModal
        report={selectedReportModal}
        onClose={() => setSelectedReportModal(null)}
        onViewOnMap={handleViewReportOnMap}
        isOfficer={true}
        onUpdateStatus={handleIncidentStatusChange}
      />
    </div>
  );
};
