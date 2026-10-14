import React, { useState } from 'react';
import { Eye, EyeOff, ExternalLink } from 'lucide-react';
import type { WasteDetection } from '../types';

interface DetectionOverlayProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
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

export const DetectionOverlay: React.FC<DetectionOverlayProps> = ({
  imageUrl,
  imageWidth,
  imageHeight,
  detections,
  inferenceMode = 'YOLOv8',
}) => {
  const [showBoxes, setShowBoxes] = useState<boolean>(true);
  const safeWidth = imageWidth > 0 ? imageWidth : 640;
  const safeHeight = imageHeight > 0 ? imageHeight : 480;

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
              const palette = CATEGORY_COLORS[det.class_name] || CATEGORY_COLORS['Plastic Waste'];
              const confPercent = Math.round(det.confidence * 100);
              const labelText = `${det.class_name} ${confPercent}%`;
              const labelWidth = Math.min(safeWidth - x1, Math.max(130, labelText.length * 9.5));
              const labelHeight = Math.max(22, Math.round(safeHeight * 0.048));
              const labelY = y1 >= labelHeight + 4 ? y1 - labelHeight : y1;

              return (
                <g key={`${det.class_name}-${idx}`}>
                  <rect
                    x={x1}
                    y={y1}
                    width={boxW}
                    height={boxH}
                    fill={palette.fill}
                    stroke={palette.stroke}
                    strokeWidth={Math.max(2.5, safeWidth * 0.004)}
                    rx={4}
                  />
                  <rect
                    x={x1}
                    y={labelY}
                    width={labelWidth}
                    height={labelHeight}
                    fill={palette.bg}
                    rx={3}
                  />
                  <text
                    x={x1 + 7}
                    y={labelY + labelHeight * 0.72}
                    fill="#FFFFFF"
                    fontSize={Math.max(12, Math.round(safeHeight * 0.028))}
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
