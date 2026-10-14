import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Cpu,
  Info,
  Layers,
} from 'lucide-react';
import type { CategoryStat, SeverityTier, WasteDetection } from '../types';

interface SeverityCardProps {
  severityScore: number;
  severityTier: SeverityTier;
  categoryCounts: Record<string, number>;
  categoryStats?: Record<string, CategoryStat>;
  detections?: WasteDetection[];
  inferenceMode: 'YOLOv8' | 'Analytical Fallback';
}

const CATEGORY_BAR_STYLES: Record<string, { bar: string; dot: string }> = {
  'Plastic Waste': { bar: 'bg-blue-600', dot: 'bg-blue-600' },
  'Organic Waste': { bar: 'bg-orange-600', dot: 'bg-orange-600' },
  'Dry Leaves / Biomass': { bar: 'bg-emerald-600', dot: 'bg-emerald-600' },
};

export const SeverityCard: React.FC<SeverityCardProps> = ({
  severityScore,
  severityTier,
  categoryCounts,
  categoryStats,
  detections = [],
  inferenceMode,
}) => {
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [showFormulaInfo, setShowFormulaInfo] = useState<boolean>(false);

  const tierStyles: Record<
    SeverityTier,
    { badge: string; bar: string; border: string; text: string }
  > = {
    High: {
      badge: 'bg-red-600 text-white',
      bar: 'bg-red-600',
      border: 'border-red-200 bg-red-50/40',
      text: 'text-red-700',
    },
    Medium: {
      badge: 'bg-amber-500 text-slate-950',
      bar: 'bg-amber-500',
      border: 'border-amber-200 bg-amber-50/40',
      text: 'text-amber-700',
    },
    Low: {
      badge: 'bg-emerald-600 text-white',
      bar: 'bg-emerald-600',
      border: 'border-emerald-200 bg-emerald-50/40',
      text: 'text-emerald-700',
    },
  };

  const activeStyle = tierStyles[severityTier] || tierStyles.Low;
  const clampedScore = Math.min(100, Math.max(0, Math.round(severityScore * 10) / 10));
  const entries = Object.entries(categoryCounts);

  const getCategoryConfidence = (cat: string): number => {
    if (categoryStats && categoryStats[cat]) {
      return Math.round(categoryStats[cat].avg_confidence * 100);
    }
    const matching = detections.filter((d) => d.class_name === cat);
    if (matching.length > 0) {
      const avg = matching.reduce((acc, d) => acc + d.confidence, 0) / matching.length;
      return Math.round(avg * 100);
    }
    return 88;
  };

  return (
    <div className="space-y-4 rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div className="flex items-center gap-2">
          <Layers className="h-5 w-5 text-emerald-600" />
          <h3 className="text-base font-bold text-slate-900">AI Detection Breakdown</h3>
        </div>
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold ${
            inferenceMode === 'YOLOv8'
              ? 'bg-emerald-100 text-emerald-800'
              : 'bg-amber-100 text-amber-800'
          }`}
        >
          <Cpu className="h-3.5 w-3.5" />
          {inferenceMode === 'YOLOv8' ? 'AI Mode (YOLOv8)' : 'Analytical Fallback Mode'}
        </span>
      </div>

      {/* Functional Waste Category Cards with Confidence Bars & Expandable Details */}
      <div className="space-y-2.5">
        {entries.length > 0 ? (
          entries.map(([category, count]) => {
            const avgConfPct = getCategoryConfidence(category);
            const catStyle =
              CATEGORY_BAR_STYLES[category] || CATEGORY_BAR_STYLES['Plastic Waste'];
            const isExpanded = expandedCategory === category;
            const catDetections = detections.filter((d) => d.class_name === category);

            return (
              <div
                key={category}
                className="overflow-hidden rounded-lg border border-slate-200/80 bg-slate-50 transition"
              >
                <button
                  type="button"
                  onClick={() =>
                    setExpandedCategory(isExpanded ? null : category)
                  }
                  className="flex w-full flex-col gap-2 px-3.5 py-2.5 text-left hover:bg-slate-100/80"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`h-2.5 w-2.5 rounded-full ${catStyle.dot}`} />
                      <span className="text-sm font-semibold text-slate-900">
                        {category}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="rounded-md bg-white px-2.5 py-0.5 text-xs font-bold text-slate-900 shadow-sm ring-1 ring-slate-200">
                        {count} {count === 1 ? 'object' : 'objects'}
                      </span>
                      {catDetections.length > 0 &&
                        (isExpanded ? (
                          <ChevronUp className="h-4 w-4 text-slate-400" />
                        ) : (
                          <ChevronDown className="h-4 w-4 text-slate-400" />
                        ))}
                    </div>
                  </div>

                  {/* Confidence Bar */}
                  <div className="flex items-center gap-2.5">
                    <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-200">
                      <div
                        className={`h-full rounded-full ${catStyle.bar}`}
                        style={{ width: `${Math.min(100, Math.max(10, avgConfPct))}%` }}
                      />
                    </div>
                    <span className="text-[11px] font-semibold text-slate-600">
                      Avg Confidence: {avgConfPct}%
                    </span>
                  </div>
                </button>

                {isExpanded && catDetections.length > 0 && (
                  <div className="border-t border-slate-200/70 bg-white px-3.5 py-2 text-xs">
                    <p className="mb-1.5 font-semibold text-slate-500">
                      Detected Bounding Boxes ({catDetections.length}):
                    </p>
                    <div className="max-h-32 space-y-1 overflow-y-auto pr-1">
                      {catDetections.map((det, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between rounded bg-slate-50 px-2 py-1 text-[11px] text-slate-700"
                        >
                          <span>
                            #{idx + 1} [{det.bbox.x1}, {det.bbox.y1} → {det.bbox.x2},{' '}
                            {det.bbox.y2}]
                          </span>
                          <span className="font-bold text-emerald-700">
                            {Math.round(det.confidence * 100)}% conf
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })
        ) : (
          <div className="rounded-lg bg-slate-50 px-3.5 py-2.5 text-sm text-slate-500">
            No distinct waste objects isolated
          </div>
        )}
      </div>

      {/* Severity Progress & Tier */}
      <div className={`rounded-xl border p-4 ${activeStyle.border}`}>
        <div className="mb-2 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
              Severity Index
            </span>
            <button
              type="button"
              onClick={() => setShowFormulaInfo((prev) => !prev)}
              className="rounded-full p-0.5 text-slate-500 hover:bg-slate-200/60 hover:text-slate-800"
              title="How is Severity calculated?"
            >
              <Info className="h-3.5 w-3.5" />
            </button>
          </div>
          <span
            className={`inline-flex items-center gap-1 rounded-md px-2.5 py-0.5 text-xs font-extrabold uppercase tracking-wide ${activeStyle.badge}`}
          >
            {severityTier === 'High' ? (
              <AlertTriangle className="h-3.5 w-3.5" />
            ) : (
              <CheckCircle2 className="h-3.5 w-3.5" />
            )}
            {severityTier.toUpperCase()}
          </span>
        </div>

        <div className="my-2 h-3 w-full overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full transition-all duration-300 ${activeStyle.bar}`}
            style={{ width: `${Math.max(4, clampedScore)}%` }}
          />
        </div>

        <div className="mt-2 flex items-baseline justify-between">
          <div className="flex items-baseline gap-1.5">
            <span className={`text-2xl font-black ${activeStyle.text}`}>
              {clampedScore} / 100
            </span>
            <span className="text-xs font-semibold text-slate-500">
              ({clampedScore}%)
            </span>
          </div>
          <span className="text-xs text-slate-500">
            Hazard-Weighted Area Ratio Index
          </span>
        </div>

        {showFormulaInfo && (
          <div className="mt-3 rounded-lg border border-slate-200 bg-white/90 p-2.5 text-xs leading-relaxed text-slate-600">
            Severity is calculated from detected waste quantity, category hazard
            weighting (Organic=1.2x, Plastic=1.0x, Dry Leaves/Biomass=0.7x),
            affected area ratio, and nearby repeat reports.
          </div>
        )}
      </div>
    </div>
  );
};
