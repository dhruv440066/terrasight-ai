import React from 'react';
import {
  Activity,
  CheckCircle2,
  Clock,
  ExternalLink,
  FileText,
  Layers,
  MapPin,
  ShieldCheck,
  Truck,
  User,
  X,
} from 'lucide-react';
import { resolveImageUrl } from '../api';
import type { IncidentRecord, IncidentStatus } from '../types';
import { DetectionOverlay } from './DetectionOverlay';

interface ReportDetailModalProps {
  report: IncidentRecord | null;
  onClose: () => void;
  onViewOnMap?: (lat: number, lon: number, hotspotId?: number | null) => void;
  isOfficer?: boolean;
  onUpdateStatus?: (reportId: number | string, newStatus: IncidentStatus) => void;
}

export const ReportDetailModal: React.FC<ReportDetailModalProps> = ({
  report,
  onClose,
  onViewOnMap,
  isOfficer = false,
  onUpdateStatus,
}) => {
  if (!report) return null;

  const fullImageUrl = resolveImageUrl(report.image_url || report.image_path);
  const categoryCounts =
    report.category_counts && Object.keys(report.category_counts).length > 0
      ? report.category_counts
      : report.detected_categories.reduce<Record<string, number>>((acc, cat) => {
          acc[cat] = (acc[cat] || 0) + 1;
          return acc;
        }, {});

  const statusBadgeColor = (status: string) => {
    switch (status) {
      case 'RESOLVED':
        return 'bg-emerald-600 text-white';
      case 'IN_PROGRESS':
        return 'bg-blue-600 text-white';
      case 'DISPATCHED':
        return 'bg-indigo-600 text-white';
      case 'ACKNOWLEDGED':
      case 'VERIFIED':
        return 'bg-amber-500 text-slate-950';
      default:
        return 'bg-slate-700 text-white';
    }
  };

  const severityBadgeColor = (tier: string) => {
    if (tier === 'High') return 'bg-red-600 text-white';
    if (tier === 'Medium') return 'bg-amber-500 text-slate-950';
    return 'bg-emerald-600 text-white';
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/75 p-4 backdrop-blur-sm">
      <div className="relative flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border border-slate-700 bg-white shadow-2xl">
        {/* Modal Header */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-slate-900 px-6 py-4 text-white">
          <div className="flex flex-wrap items-center gap-3">
            <span className="rounded-lg bg-emerald-600 px-3 py-1 text-sm font-black tracking-wide text-white">
              {report.incident_uuid}
            </span>
            <span
              className={`rounded-md px-2.5 py-1 text-xs font-extrabold uppercase tracking-wider ${statusBadgeColor(
                report.status
              )}`}
            >
              {report.status}
            </span>
            <span
              className={`rounded-md px-2.5 py-1 text-xs font-extrabold uppercase ${severityBadgeColor(
                report.severity_tier
              )}`}
            >
              {report.severity_tier} ({report.severity_score} / 100)
            </span>
            {report.duplicate_status && (
              <span
                className={`rounded-md px-2.5 py-1 text-[11px] font-bold uppercase ${
                  report.is_primary !== false
                    ? 'bg-emerald-500/20 text-emerald-300 ring-1 ring-emerald-500/40'
                    : 'bg-amber-500/20 text-amber-300 ring-1 ring-amber-500/40'
                }`}
              >
                {report.duplicate_status}
              </span>
            )}
          </div>

          <button
            type="button"
            onClick={onClose}
            className="rounded-lg bg-slate-800 p-1.5 text-slate-300 transition hover:bg-slate-700 hover:text-white"
            aria-label="Close report detail modal"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="grid flex-1 grid-cols-1 gap-6 overflow-y-auto p-6 lg:grid-cols-12">
          {/* Left Column: Image + Bounding Boxes + Detection Details */}
          <div className="space-y-4 lg:col-span-7">
            <DetectionOverlay
              imageUrl={fullImageUrl}
              imageWidth={640}
              imageHeight={480}
              detections={report.detections || []}
              inferenceMode="YOLOv8"
            />

            <div className="flex flex-wrap items-center justify-between gap-2">
              <a
                href={fullImageUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-xs font-bold text-slate-700 shadow-sm hover:bg-slate-50"
              >
                <ExternalLink className="h-3.5 w-3.5 text-emerald-600" />
                Open Full Image
              </a>

              {onViewOnMap && (
                <button
                  type="button"
                  onClick={() => {
                    onViewOnMap(report.latitude, report.longitude, report.hotspot_id);
                    onClose();
                  }}
                  className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-3.5 py-1.5 text-xs font-bold text-white shadow-sm hover:bg-emerald-700"
                >
                  <MapPin className="h-3.5 w-3.5" />
                  View On Map
                </button>
              )}
            </div>

            {/* AI Detections Summary */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
              <div className="mb-2.5 flex items-center justify-between">
                <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
                  <Layers className="h-4 w-4 text-emerald-600" />
                  AI Waste Detections ({report.detection_count} Objects)
                </span>
                {report.average_confidence !== undefined && (
                  <span className="text-xs font-semibold text-emerald-700">
                    Avg Confidence: {Math.round(report.average_confidence * 100)}%
                  </span>
                )}
              </div>

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                {Object.entries(categoryCounts).map(([cat, cnt]) => (
                  <div
                    key={cat}
                    className="rounded-lg border border-slate-200 bg-white p-2.5 shadow-sm"
                  >
                    <span className="block text-xs font-semibold text-slate-600">
                      {cat}
                    </span>
                    <span className="mt-0.5 block text-sm font-black text-slate-900">
                      {cnt} {cnt === 1 ? 'object' : 'objects'}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {report.description && (
              <div className="rounded-xl border border-slate-200 bg-white p-3.5">
                <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-500">
                  <FileText className="h-3.5 w-3.5 text-slate-400" />
                  Citizen Notes / Description
                </span>
                <p className="mt-1 text-sm text-slate-800">{report.description}</p>
              </div>
            )}
          </div>

          {/* Right Column: Metadata, Location, Hotspot, Dispatch & Audit Trail */}
          <div className="space-y-4 lg:col-span-5">
            {/* Reporter & Metadata Card */}
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-2.5 text-xs">
              <div className="flex items-center justify-between border-b border-slate-200 pb-2">
                <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-slate-600">
                  <User className="h-3.5 w-3.5 text-emerald-600" />
                  Citizen Report Metadata
                </span>
                <span className="font-mono font-semibold text-slate-500">
                  ID #{report.id}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <span className="block text-[11px] text-slate-500">Reported By</span>
                  <span className="font-bold text-slate-900">
                    {report.citizen_name || report.citizen_username}
                  </span>
                </div>
                <div>
                  <span className="block text-[11px] text-slate-500">Submitted At</span>
                  <span className="font-semibold text-slate-800">
                    {new Date(report.created_at).toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="block text-[11px] text-slate-500">Hotspot Cluster</span>
                  <span className="font-bold text-emerald-700">
                    {report.hotspot_code ||
                      (report.hotspot_id ? `HS-${String(report.hotspot_id).padStart(4, '0')}` : 'Unassigned')}
                  </span>
                </div>
                <div>
                  <span className="block text-[11px] text-slate-500">Cluster Offset</span>
                  <span className="font-semibold text-slate-800">
                    {report.distance_from_centroid !== undefined
                      ? `${report.distance_from_centroid}m from centroid`
                      : '0.0m (Centroid)'}
                  </span>
                </div>
              </div>
            </div>

            {/* Geolocation & Reverse-Geocoded Address */}
            <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-slate-600">
                  <MapPin className="h-3.5 w-3.5 text-emerald-600" />
                  Verified Geolocation
                </span>
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-[10px] font-bold text-emerald-800">
                  {report.location_source}
                </span>
              </div>
              <p className="text-xs font-semibold text-slate-900">
                {report.address ||
                  `Municipal Coordinate (${report.latitude.toFixed(4)}°N, ${report.longitude.toFixed(4)}°E)`}
              </p>
              <p className="font-mono text-[11px] text-slate-500">
                Lat: {report.latitude.toFixed(6)} | Lon: {report.longitude.toFixed(6)}
              </p>
            </div>

            {/* Assigned Dispatches */}
            {report.dispatches && report.dispatches.length > 0 && (
              <div className="rounded-xl border border-indigo-200 bg-indigo-50/50 p-4 space-y-2 text-xs">
                <span className="flex items-center gap-1.5 font-bold uppercase tracking-wider text-indigo-900">
                  <Truck className="h-4 w-4 text-indigo-600" />
                  Sanitation Crew Dispatch
                </span>
                {report.dispatches.map((d) => (
                  <div
                    key={d.id}
                    className="rounded-lg border border-indigo-100 bg-white p-2.5 shadow-sm"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900">{d.crew_name}</span>
                      <span className="rounded bg-indigo-100 px-2 py-0.5 text-[10px] font-bold text-indigo-800">
                        {d.status}
                      </span>
                    </div>
                    {d.vehicle_number && (
                      <p className="mt-0.5 text-[11px] font-semibold text-slate-600">
                        Vehicle: {d.vehicle_number}
                      </p>
                    )}
                    {d.notes && <p className="mt-1 text-[11px] text-slate-600">{d.notes}</p>}
                  </div>
                ))}
              </div>
            )}

            {/* Officer Quick Status Actions */}
            {isOfficer && onUpdateStatus && (
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 space-y-2">
                <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
                  <ShieldCheck className="h-4 w-4 text-emerald-600" />
                  Officer Status Control
                </span>
                <div className="grid grid-cols-2 gap-2 pt-1">
                  {(
                    [
                      'ACKNOWLEDGED',
                      'DISPATCHED',
                      'IN_PROGRESS',
                      'RESOLVED',
                    ] as IncidentStatus[]
                  ).map((st) => (
                    <button
                      key={st}
                      type="button"
                      onClick={() => onUpdateStatus(report.id, st)}
                      className={`rounded-lg px-2.5 py-1.5 text-xs font-bold transition ${
                        report.status === st
                          ? 'bg-emerald-600 text-white shadow-sm'
                          : 'border border-slate-300 bg-white text-slate-700 hover:bg-slate-100'
                      }`}
                    >
                      {st.replace('_', ' ')}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Audit Trail Timeline */}
            <div className="rounded-xl border border-slate-200 bg-white p-4 space-y-3">
              <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wider text-slate-700">
                <Activity className="h-4 w-4 text-emerald-600" />
                Audit Trail & Status History
              </span>

              {report.audit_logs && report.audit_logs.length > 0 ? (
                <div className="max-h-48 space-y-2.5 overflow-y-auto pr-1">
                  {report.audit_logs.map((log) => (
                    <div
                      key={log.id}
                      className="relative border-l-2 border-emerald-500 pl-3 text-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-slate-900">
                          {log.action.replace(/_/g, ' ')}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          {log.timestamp
                            ? new Date(log.timestamp).toLocaleTimeString([], {
                                hour: '2-digit',
                                minute: '2-digit',
                              })
                            : ''}
                        </span>
                      </div>
                      <p className="mt-0.5 text-[11px] text-slate-600">{log.notes}</p>
                      <span className="text-[10px] font-semibold text-emerald-700">
                        {log.actor_name} ({log.actor_role})
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-2 text-xs text-slate-600">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600" />
                    <span>
                      Report {report.incident_uuid} created ({report.status})
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-slate-400">
                    <Clock className="h-3.5 w-3.5" />
                    <span>Updated {new Date(report.updated_at).toLocaleString()}</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
