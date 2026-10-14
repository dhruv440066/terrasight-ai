import React, { useState } from 'react';
import { AlertCircle, Send, Truck, X } from 'lucide-react';
import type { HotspotRecord } from '../types';
import { formatCoordinates } from '../utils/geoCalculations';

interface DispatchModalProps {
  hotspot: HotspotRecord | null;
  isOpen: boolean;
  onClose: () => void;
  onConfirmDispatch: (hotspotId: number, crewName: string, notes: string) => Promise<void>;
}

const PRESET_CREWS = [
  'North Zone Sanitation Team A',
  'Central Sector Heavy Waste Team B',
  'East Ward Compactor Crew C',
  'West Zone Bio-Sanitation Squad D',
  'South Sector Green Sweeper Team E',
];

export const DispatchModal: React.FC<DispatchModalProps> = ({
  hotspot,
  isOpen,
  onClose,
  onConfirmDispatch,
}) => {
  const [crewName, setCrewName] = useState<string>('North Zone Sanitation Team A');
  const [notes, setNotes] = useState<string>(
    'Large plastic accumulation near roadside.'
  );
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen || !hotspot) {
    return null;
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!crewName.trim()) {
      setError('Please enter a sanitation crew name.');
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      await onConfirmDispatch(hotspot.id, crewName.trim(), notes.trim());
      onClose();
    } catch (err: unknown) {
      const msg =
        err instanceof Error ? err.message : 'Failed to dispatch sanitation crew.';
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-xs">
      <div className="w-full max-w-lg overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-center justify-between bg-[#0F172A] px-6 py-4 text-white">
          <div className="flex items-center gap-2.5">
            <Truck className="h-5 w-5 text-emerald-400" />
            <h3 className="text-base font-bold">
              Dispatch Sanitation Crew — Hotspot #{hotspot.id}
            </h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-slate-300 hover:bg-slate-800 hover:text-white"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 p-6">
          {/* Hotspot Summary */}
          <div className="grid grid-cols-3 gap-2 rounded-xl border border-slate-200 bg-slate-50 p-3 text-xs">
            <div>
              <span className="block text-slate-500">Priority / Severity</span>
              <span className="font-extrabold text-slate-900">
                {hotspot.severity_tier.toUpperCase()} ({hotspot.maximum_severity}%)
              </span>
            </div>
            <div>
              <span className="block text-slate-500">Clustered Reports</span>
              <span className="font-extrabold text-slate-900">
                {hotspot.report_count} ({hotspot.duplicate_reports_suppressed} deduped)
              </span>
            </div>
            <div>
              <span className="block text-slate-500">Coordinates</span>
              <span className="font-bold text-slate-900">
                {formatCoordinates(hotspot.latitude, hotspot.longitude, 4)}
              </span>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs font-medium text-red-700">
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Crew Name Field */}
          <div>
            <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
              Crew Name
            </label>
            <input
              type="text"
              required
              value={crewName}
              onChange={(e) => setCrewName(e.target.value)}
              placeholder="e.g. North Zone Sanitation Team A"
              className="w-full rounded-lg border border-slate-300 px-3.5 py-2.5 text-sm text-slate-900 focus:border-emerald-600 focus:outline-none"
            />
            <div className="mt-2 flex flex-wrap gap-1.5">
              {PRESET_CREWS.map((preset) => (
                <button
                  key={preset}
                  type="button"
                  onClick={() => setCrewName(preset)}
                  className={`rounded-md px-2 py-1 text-[11px] font-medium transition ${
                    crewName === preset
                      ? 'bg-emerald-600 text-white'
                      : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                  }`}
                >
                  {preset}
                </button>
              ))}
            </div>
          </div>

          {/* Optional Notes */}
          <div>
            <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
              Optional Notes
            </label>
            <textarea
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Enter dispatch instructions, equipment needed, or site notes..."
              className="w-full rounded-lg border border-slate-300 px-3.5 py-2 text-sm text-slate-900 focus:border-emerald-600 focus:outline-none"
            />
          </div>

          <div className="flex items-center justify-end gap-3 border-t border-slate-100 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-5 py-2 text-sm font-bold text-white shadow-xs hover:bg-emerald-700 disabled:opacity-50"
            >
              <Send className="h-4 w-4" />
              {submitting ? 'Dispatching...' : 'Dispatch Sanitation Crew'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
