import React from 'react';
import {
  Calendar,
  CheckCircle2,
  Circle,
  Clock,
  Eye,
  MapPin,
  Truck,
} from 'lucide-react';
import { resolveImageUrl } from '../api';
import { formatCoordinates } from '../utils/geoCalculations';
import type { IncidentRecord, IncidentStatus, SeverityTier } from '../types';

interface IncidentCardProps {
  incident: IncidentRecord;
  onSelect?: (incident: IncidentRecord) => void;
}

const TIMELINE_STEPS: { key: IncidentStatus; label: string }[] = [
  { key: 'SUBMITTED', label: 'Submitted' },
  { key: 'ACKNOWLEDGED', label: 'Acknowledged' },
  { key: 'DISPATCHED', label: 'Dispatched' },
  { key: 'IN_PROGRESS', label: 'In Progress' },
  { key: 'RESOLVED', label: 'Resolved' },
];

const STATUS_ORDER: Record<IncidentStatus, number> = {
  SUBMITTED: 0,
  PENDING: 0,
  VERIFIED: 1,
  ACKNOWLEDGED: 1,
  DISPATCHED: 2,
  IN_PROGRESS: 3,
  RESOLVED: 4,
  REJECTED: -1,
};

export const IncidentCard: React.FC<IncidentCardProps> = ({
  incident,
  onSelect,
}) => {
  const currentStepIndex = STATUS_ORDER[incident.status] ?? 0;

  const tierColors: Record<SeverityTier, string> = {
    High: 'bg-red-100 text-red-800 border-red-200',
    Medium: 'bg-amber-100 text-amber-800 border-amber-200',
    Low: 'bg-emerald-100 text-emerald-800 border-emerald-200',
  };

  const formattedDate = incident.created_at
    ? new Date(incident.created_at).toLocaleDateString('en-GB', {
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      })
    : 'Today';

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm transition hover:shadow-md">
      <div className="grid grid-cols-1 gap-4 p-5 md:grid-cols-12">
        {/* Image Preview */}
        <div className="md:col-span-4">
          <div className="relative h-44 w-full overflow-hidden rounded-lg bg-slate-100">
            <img
              src={resolveImageUrl(incident.image_url || incident.image_path)}
              alt={`Incident ${incident.incident_uuid}`}
              className="h-full w-full object-cover"
            />
            <span className="absolute bottom-2 left-2 rounded bg-slate-900/80 px-2 py-0.5 text-[11px] font-semibold text-white">
              {incident.location_source}
            </span>
            {incident.duplicate_status && (
              <span
                className={`absolute right-2 top-2 rounded px-2 py-0.5 text-[10px] font-bold uppercase ${
                  incident.is_primary !== false
                    ? 'bg-emerald-600/90 text-white'
                    : 'bg-amber-500/90 text-slate-950'
                }`}
              >
                {incident.duplicate_status}
              </span>
            )}
          </div>
        </div>

        {/* Metadata */}
        <div className="flex flex-col justify-between space-y-3 md:col-span-5">
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h4 className="text-lg font-extrabold text-slate-900">
                Report {incident.incident_uuid}
              </h4>
              <span
                className={`rounded-full border px-2.5 py-0.5 text-xs font-bold uppercase ${
                  tierColors[incident.severity_tier] || tierColors.Low
                }`}
              >
                {incident.severity_tier.toUpperCase()} ({incident.severity_score} / 100)
              </span>
            </div>

            <div className="mt-2 space-y-1.5 text-xs text-slate-600">
              <div className="flex items-center gap-1.5">
                <Calendar className="h-3.5 w-3.5 text-slate-400" />
                <span>
                  Submitted: <strong className="text-slate-800">{formattedDate}</strong>
                </span>
              </div>
              <div className="flex items-start gap-1.5">
                <MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-600" />
                <span>
                  Location:{' '}
                  <strong className="text-slate-800">
                    {incident.address ||
                      formatCoordinates(incident.latitude, incident.longitude, 4)}
                  </strong>
                  <span className="ml-1 font-mono text-[11px] text-slate-500">
                    ({formatCoordinates(incident.latitude, incident.longitude, 4)})
                  </span>
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-1.5">
                <Clock className="h-3.5 w-3.5 text-slate-400" />
                <span>
                  Status:{' '}
                  <strong className="rounded bg-slate-100 px-1.5 py-0.5 text-slate-900">
                    {incident.status}
                  </strong>
                </span>
                {incident.hotspot_id && (
                  <span className="ml-1 rounded bg-emerald-50 px-1.5 py-0.5 text-[11px] font-semibold text-emerald-700">
                    {incident.hotspot_code || `Hotspot #${incident.hotspot_id}`}
                  </span>
                )}
              </div>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
            <div className="flex flex-wrap gap-1.5">
              {incident.detected_categories.map((cat) => (
                <span
                  key={cat}
                  className="rounded-md bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-700"
                >
                  {cat}
                </span>
              ))}
            </div>

            {onSelect && (
              <button
                type="button"
                onClick={() => onSelect(incident)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600/30 bg-emerald-50 px-3 py-1.5 text-xs font-bold text-emerald-700 transition hover:bg-emerald-600 hover:text-white"
              >
                <Eye className="h-3.5 w-3.5" />
                View Full Report
              </button>
            )}
          </div>

          {incident.dispatches && incident.dispatches.length > 0 && (
            <div className="rounded-lg border border-emerald-200 bg-emerald-50/60 px-3 py-2 text-xs text-emerald-900">
              <div className="flex items-center gap-1.5 font-bold">
                <Truck className="h-3.5 w-3.5 text-emerald-700" />
                <span>Assigned Crew: {incident.dispatches[0].crew_name}</span>
              </div>
              {incident.dispatches[0].notes && (
                <p className="mt-0.5 text-[11px] text-emerald-800">
                  {incident.dispatches[0].notes}
                </p>
              )}
            </div>
          )}
        </div>

        {/* Status Timeline */}
        <div className="border-t border-slate-100 pt-3 md:col-span-3 md:border-l md:border-t-0 md:pl-4 md:pt-0">
          <span className="mb-2 block text-[11px] font-bold uppercase tracking-wider text-slate-400">
            Status Timeline
          </span>
          <div className="space-y-2">
            {TIMELINE_STEPS.map((step, index) => {
              const isCompleted = currentStepIndex >= index;
              const isCurrent = currentStepIndex === index;
              return (
                <div key={step.key} className="flex items-center gap-2.5 text-xs">
                  {isCompleted ? (
                    <CheckCircle2
                      className={`h-4 w-4 shrink-0 ${
                        isCurrent ? 'text-emerald-600' : 'text-slate-700'
                      }`}
                    />
                  ) : (
                    <Circle className="h-4 w-4 shrink-0 text-slate-300" />
                  )}
                  <span
                    className={
                      isCurrent
                        ? 'font-bold text-emerald-700'
                        : isCompleted
                        ? 'font-semibold text-slate-800'
                        : 'text-slate-400'
                    }
                  >
                    {step.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
