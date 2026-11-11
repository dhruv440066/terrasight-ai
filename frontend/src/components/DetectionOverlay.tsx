import React, { useState } from 'react';
import { Eye, EyeOff, ExternalLink } from 'lucide-react';
import type { WasteDetection } from '../types';

interface DetectionOverlayProps {
  imageUrl: string;
  imageWidth?: number;
  imageHeight?: number;
  detections: WasteDetection[];
  inferenceMode?: 'YOLOv8' | 'Analytical Fallback';
}

const CATEGORY_COLORS: Record<string, { stroke: string; fill: string; bg: string }> = {
  'Plastic Waste': {
    stroke: '#2563EB',
    fill: 'rgba(37, 99, 235, 0.14)',
    bg: '#1D4ED8',
  },
  'Organic Waste': {
    stroke: '#EA580C',
    fill: 'rgba(234, 88, 12, 0.14)',
    bg: '#C2410C',
  },
  'Dry Leaves / Biomass': {
    stroke: '#16A34A',
    fill: 'rgba(22, 163, 74, 0.14)',
    bg: '#15803D',
  },
};

const SUBCLASS_COLORS: Record<string, { stroke: string; fill: string; bg: string }> = {
  'Plastic bottle': { stroke: '#1D4ED8', fill: 'rgba(29, 78, 216, 0.14)', bg: '#1E40AF' },
  'Clear plastic bottle': { stroke: '#0284C7', fill: 'rgba(2, 132, 199, 0.14)', bg: '#0369A1' },
  'Other plastic bottle': { stroke: '#2563EB', fill: 'rgba(37, 99, 235, 0.14)', bg: '#1D4ED8' },
  'Glass bottle & jar': { stroke: '#DC2626', fill: 'rgba(220, 38, 38, 0.13)', bg: '#B91C1C' },
  'Drink & metal can': { stroke: '#059669', fill: 'rgba(5, 150, 105, 0.13)', bg: '#047857' },
  'Plastic bag & film': { stroke: '#7C3AED', fill: 'rgba(124, 58, 237, 0.14)', bg: '#6D28D9' },
  'Garbage bag': { stroke: '#EA580C', fill: 'rgba(234, 88, 12, 0.14)', bg: '#C2410C' },
  'Plastic wrapper': { stroke: '#0891B2', fill: 'rgba(8, 145, 178, 0.14)', bg: '#0E7490' },
  'Disposable cup': { stroke: '#4F46E5', fill: 'rgba(79, 70, 229, 0.14)', bg: '#4338CA' },
  'Styrofoam piece': { stroke: '#DB2777', fill: 'rgba(219, 39, 119, 0.14)', bg: '#BE185D' },
  'Bottle cap, straw & lid': { stroke: '#16A34A', fill: 'rgba(22, 163, 74, 0.14)', bg: '#15803D' },
  'Food & takeout container': { stroke: '#D97706', fill: 'rgba(217, 119, 6, 0.14)', bg: '#B45309' },
  'Food & organic waste': { stroke: '#EA580C', fill: 'rgba(234, 88, 12, 0.15)', bg: '#C2410C' },
  'Cardboard carton': { stroke: '#9333EA', fill: 'rgba(147, 51, 234, 0.13)', bg: '#7E22CE' },
  'Paper & paper bag': { stroke: '#0D9488', fill: 'rgba(13, 148, 136, 0.13)', bg: '#0F766E' },
  'Dry leaves & foliage': { stroke: '#16A34A', fill: 'rgba(22, 163, 74, 0.15)', bg: '#15803D' },
  'Cigarette': { stroke: '#CA8A04', fill: 'rgba(202, 138, 4, 0.15)', bg: '#A16207' },
  'Mixed litter': { stroke: '#475569', fill: 'rgba(71, 85, 105, 0.13)', bg: '#334155' },
};

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  imageUrl,
  imageWidth = 0,
  imageHeight = 0,
  detections,
  inferenceMode = 'YOLOv8',
}) => {
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number } | null>(null);

  const fallbackW = naturalDims
    ? naturalDims.w > 1024 || naturalDims.h > 1024
      ? Math.round((naturalDims.w * 1024) / Math.max(naturalDims.w, naturalDims.h))
      : naturalDims.w
    : 640;
  const fallbackH = naturalDims
    ? naturalDims.w > 1024 || naturalDims.h > 1024
      ? Math.round((naturalDims.h * 1024) / Math.max(naturalDims.w, naturalDims.h))
      : naturalDims.h
    : 480;

  const safeWidth = imageWidth && imageWidth > 0 ? imageWidth : fallbackW;
  const safeHeight = imageHeight && imageHeight > 0 ? imageHeight : fallbackH;

  return (
    <div className="relative w-full overflow-hidden rounded-xl border border-slate-200 bg-slate-950 shadow-sm">
      {/* Top Bar Controls: Mode Badge + Bounding Box Toggle + Open Full Image */}
      <div className="absolute inset-x-3 top-3 z-20 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
        <div className="pointer-events-auto flex items-center gap-2 rounded-full bg-slate-900/85 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm">
          <span
            className={`h-2 w-2 rounded-full ${
              inferenceMode === 'YOLOv8' ? 'bg-emerald-400' : 'bg-amber-400'
            }`}
          />
          <span>
            {inferenceMode === 'YOLOv8' ? 'AI Mode (YOLOv8)' : 'Analytical Fallback Mode'}
          </span>
          <span className="ml-1 rounded bg-white/15 px-1.5 py-0.5 text-[10px] font-bold">
            {detections.length} {detections.length === 1 ? 'box' : 'boxes'}
          </span>
        </div>

        <div className="pointer-events-auto flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setShowBoxes((prev) => !prev)}
            className="inline-flex items-center gap-1.5 rounded-full bg-slate-900/85 px-3 py-1 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-slate-800"
            title={showBoxes ? 'Hide bounding boxes' : 'Show bounding boxes'}
          >
            {showBoxes ? (
              <>
                <EyeOff className="h-3.5 w-3.5 text-emerald-400" />
                <span>Hide Boxes</span>
              </>
            ) : (
              <>
                <Eye className="h-3.5 w-3.5 text-slate-300" />
                <span>Show Boxes</span>
              </>
            )}
          </button>

          {imageUrl && (
            <a
              href={imageUrl}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 rounded-full bg-slate-900/85 px-2.5 py-1 text-xs font-semibold text-white backdrop-blur-sm transition hover:bg-slate-800"
              title="Open full resolution image in new tab"
            >
              <ExternalLink className="h-3.5 w-3.5 text-emerald-400" />
              <span className="hidden sm:inline">Full Image</span>
            </a>
          )}
        </div>
      </div>

      <div className="relative mx-auto w-full">
        <img
          src={imageUrl}
          alt="Uploaded civic waste preview"
          onLoad={(e) => {
            const img = e.currentTarget;
            if (img.naturalWidth > 0 && img.naturalHeight > 0) {
              setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight });
            }
          }}
          className="block max-h-[460px] w-full select-none object-contain"
        />

        {/* Non-destructive SVG Bounding Box Overlay */}
        {showBoxes && (
          <svg
            viewBox={`0 0 ${safeWidth} ${safeHeight}`}
            preserveAspectRatio="xMidYMid meet"
            className="pointer-events-none absolute inset-0 h-full w-full"
          >
            {detections.map((det, idx) => {
              const { x1, y1, x2, y2 } = det.bbox;
              const boxW = Math.max(8, x2 - x1);
              const boxH = Math.max(8, y2 - y1);
              const displayLabel = det.subclass || det.class_name;
              const palette =
                (det.subclass && SUBCLASS_COLORS[det.subclass]) ||
                CATEGORY_COLORS[det.class_name] ||
                CATEGORY_COLORS['Plastic Waste'];
              const confPercent = Math.round(det.confidence * 100);
              const labelText = `${displayLabel} ${confPercent}%`;
              const fontSize = Math.max(11, Math.min(20, Math.round(safeHeight * 0.023)));
              const labelHeight = Math.round(fontSize * 1.48);
              const labelWidth = Math.min(
                safeWidth,
                Math.max(82, Math.round(labelText.length * fontSize * 0.56 + 14))
              );
              const labelX = Math.max(0, Math.min(x1, safeWidth - labelWidth));
              const labelY = y1 >= labelHeight + 3 ? y1 - labelHeight : y1;

              return (
                <g key={`${displayLabel}-${idx}`}>
                  <rect
                    x={x1}
                    y={y1}
                    width={boxW}
                    height={boxH}
                    fill={palette.fill}
                    stroke={palette.stroke}
                    strokeWidth={Math.max(2, safeWidth * 0.0032)}
                    rx={3}
                  />
                  <rect
                    x={labelX}
                    y={labelY}
                    width={labelWidth}
                    height={labelHeight}
                    fill={palette.bg}
                    rx={3}
                  />
                  <text
                    x={labelX + 6}
                    y={labelY + labelHeight * 0.72}
                    fill="#FFFFFF"
                    fontSize={fontSize}
                    fontWeight="700"
                    fontFamily="system-ui, -apple-system, sans-serif"
                  >
                    {labelText}
                  </text>
                </g>
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
};
