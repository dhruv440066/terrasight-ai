import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Eye,
  FileText,
  Leaf,
  Loader2,
  LogOut,
  MapPin,
  RefreshCw,
  Send,
  Upload,
  User as UserIcon,
  X,
} from 'lucide-react';
import {
  analyzeWasteImageApi,
  fetchMyIncidentsApi,
  resolveImageUrl,
  submitIncidentApi,
} from '../api';
import type {
  AnalysisResponse,
  IncidentRecord,
  LocationSource,
  UserProfile,
} from '../types';
import { DetectionOverlay } from './DetectionOverlay';
import { IncidentCard } from './IncidentCard';
import { MapPicker } from './MapPicker';
import { ReportDetailModal } from './ReportDetailModal';
import { SeverityCard } from './SeverityCard';

interface CitizenViewProps {
  user: UserProfile | null;
  isGuest: boolean;
  onLogout: () => void;
}

type CitizenTab = 'report' | 'reports' | 'profile';

const GUEST_INCIDENTS_STORAGE_KEY = 'cleanpath_guest_incidents';

const ANALYSIS_PROGRESS_STEPS = [
  'Uploading waste photograph...',
  'Running YOLOv8 multi-scale waste detection...',
  'Extracting EXIF GPS metadata & reverse geocoding...',
  'Calculating hazard-weighted severity index...',
];

export const CitizenView: React.FC<CitizenViewProps> = ({
  user,
  isGuest,
  onLogout,
}) => {
  const location = useLocation();
  const navigate = useNavigate();

  const initialTab: CitizenTab = location.pathname.includes('/reports')
    ? 'reports'
    : 'report';
  const [activeTab, setActiveTab] = useState<CitizenTab>(initialTab);

  // Sync tab with route
  useEffect(() => {
    if (location.pathname.includes('/reports')) {
      setActiveTab('reports');
    } else if (location.pathname.includes('/report')) {
      setActiveTab('report');
    }
  }, [location.pathname]);

  const handleTabChange = (tab: CitizenTab) => {
    setActiveTab(tab);
    if (tab === 'report') navigate('/citizen/report');
    else if (tab === 'reports') navigate('/citizen/reports');
    else navigate('/citizen');
  };

  // Upload & Analysis State
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [analyzing, setAnalyzing] = useState<boolean>(false);
  const [analysisStepIdx, setAnalysisStepIdx] = useState<number>(0);
  const [submitting, setSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [latitude, setLatitude] = useState<number | null>(null);
  const [longitude, setLongitude] = useState<number | null>(null);
  const [locationSource, setLocationSource] = useState<LocationSource>('MANUAL');
  const [address, setAddress] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [showManualMapOverride, setShowManualMapOverride] = useState<boolean>(false);
  const [submittedIncident, setSubmittedIncident] = useState<IncidentRecord | null>(
    null
  );
  const [selectedReportModal, setSelectedReportModal] =
    useState<IncidentRecord | null>(null);

  // Camera Modal State
  const [cameraOpen, setCameraOpen] = useState<boolean>(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // My Incidents State
  const [myIncidents, setMyIncidents] = useState<IncidentRecord[]>([]);
  const [loadingIncidents, setLoadingIncidents] = useState<boolean>(false);

  // Cycle progress text while AI analysis is running
  useEffect(() => {
    if (!analyzing) {
      setAnalysisStepIdx(0);
      return;
    }
    const timer = setInterval(() => {
      setAnalysisStepIdx((prev) =>
        prev < ANALYSIS_PROGRESS_STEPS.length - 1 ? prev + 1 : prev
      );
    }, 650);
    return () => clearInterval(timer);
  }, [analyzing]);

  const loadMyIncidents = useCallback(async () => {
    setLoadingIncidents(true);
    try {
      let guestUuids: string[] | undefined;
      if (isGuest) {
        const stored = localStorage.getItem(GUEST_INCIDENTS_STORAGE_KEY);
        if (stored) {
          guestUuids = JSON.parse(stored) as string[];
        }
      }
      const list = await fetchMyIncidentsApi(guestUuids);
      setMyIncidents(list);
    } catch {
      // Ignore transient errors on initial load
    } finally {
      setLoadingIncidents(false);
    }
  }, [isGuest]);

  useEffect(() => {
    loadMyIncidents();
  }, [loadMyIncidents]);

  const stopCameraStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraOpen(false);
  };

  const handleOpenCamera = async () => {
    setError(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      fileInputRef.current?.click();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
      });
      streamRef.current = stream;
      setCameraOpen(true);
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      }, 100);
    } catch {
      fileInputRef.current?.click();
    }
  };

  const handleCapturePhoto = async () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      async (blob) => {
        stopCameraStream();
        if (!blob) return;
        const capturedFile = new File([blob], `camera_capture_${Date.now()}.jpg`, {
          type: 'image/jpeg',
        });
        await processSelectedFile(capturedFile);
      },
      'image/jpeg',
      0.92
    );
  };

  const processSelectedFile = async (file: File) => {
    setError(null);
    setSubmittedIncident(null);

    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    const ext = file.name.toLowerCase();
    if (
      !validTypes.includes(file.type) &&
      !ext.endsWith('.jpg') &&
      !ext.endsWith('.jpeg') &&
      !ext.endsWith('.png')
    ) {
      setError('Invalid file format. Please upload a .jpg, .jpeg, or .png photograph.');
      return;
    }

    setAnalyzing(true);
    try {
      const result = await analyzeWasteImageApi(file);
      setAnalysis(result);
      setAddress(result.address || '');
      if (
        result.gps.source === 'EXIF' &&
        result.gps.latitude !== null &&
        result.gps.longitude !== null
      ) {
        setLatitude(result.gps.latitude);
        setLongitude(result.gps.longitude);
        setLocationSource('EXIF');
        setShowManualMapOverride(false);
      } else {
        setLatitude(28.6139);
        setLongitude(77.209);
        setLocationSource('MANUAL');
        setShowManualMapOverride(true);
      }
    } catch (err: unknown) {
      let msg = 'Failed to analyze the waste image. Please try another photograph.';
      if (
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data
          ?.detail === 'string'
      ) {
        msg = (err as { response: { data: { detail: string } } }).response.data.detail;
      }
      setError(msg);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await processSelectedFile(file);
    }
  };

  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      await processSelectedFile(file);
    }
  };

  const handleCoordinateChange = useCallback(
    (lat: number, lon: number, source: LocationSource, resolvedAddr?: string) => {
      setLatitude(lat);
      setLongitude(lon);
      setLocationSource(source);
      if (resolvedAddr) {
        setAddress(resolvedAddr);
      }
    },
    []
  );

  const handleSubmitComplaint = async () => {
    if (!analysis) return;
    if (
      latitude === null ||
      longitude === null ||
      Number.isNaN(latitude) ||
      Number.isNaN(longitude)
    ) {
      setError('Please select valid waste coordinates on the map before submitting.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const created = await submitIncidentApi({
        image_path: analysis.image_path,
        original_filename: analysis.original_filename,
        latitude,
        longitude,
        location_source: locationSource,
        address: address || undefined,
        description: description.trim() || undefined,
        detected_categories: analysis.detected_categories,
        detections: analysis.detections,
        detection_count: analysis.detection_count,
        severity_score: analysis.severity_score,
        severity_tier: analysis.severity_tier,
      });

      if (isGuest) {
        const existingRaw = localStorage.getItem(GUEST_INCIDENTS_STORAGE_KEY);
        const existingList: string[] = existingRaw ? JSON.parse(existingRaw) : [];
        if (!existingList.includes(created.incident_uuid)) {
          existingList.unshift(created.incident_uuid);
          localStorage.setItem(
            GUEST_INCIDENTS_STORAGE_KEY,
            JSON.stringify(existingList)
          );
        }
      }

      setSubmittedIncident(created);
      await loadMyIncidents();
    } catch (err: unknown) {
      let msg = 'Failed to submit complaint. Please try again.';
      if (
        typeof err === 'object' &&
        err !== null &&
        'response' in err &&
        typeof (err as { response?: { data?: { detail?: string } } }).response?.data
          ?.detail === 'string'
      ) {
        msg = (err as { response: { data: { detail: string } } }).response.data.detail;
      }
      setError(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const handleResetReport = () => {
    setAnalysis(null);
    setSubmittedIncident(null);
    setError(null);
    setLatitude(null);
    setLongitude(null);
    setDescription('');
  };

  // Determine active step for 4-step workflow bar
  const currentStep = submittedIncident
    ? 4
    : analysis
    ? latitude !== null && longitude !== null
      ? 3
      : 2
    : analyzing
    ? 2
    : 1;

  const workflowSteps = [
    { num: '01', label: 'Upload Image', active: currentStep >= 1 },
    { num: '02', label: 'AI Analysis', active: currentStep >= 2 },
    { num: '03', label: 'Confirm Location', active: currentStep >= 3 },
    { num: '04', label: 'Submit Report', active: currentStep >= 4 },
  ];

  return (
    <div className="min-h-screen bg-[#F8FAFC] text-[#0F172A]">
      {/* Top Navigation */}
      <header className="sticky top-0 z-30 bg-[#0F172A] text-white shadow-md">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-emerald-600">
              <Leaf className="h-5 w-5 text-white" />
            </div>
            <div>
              <span className="block text-base font-black tracking-tight leading-none">
                CLEANPATH AI
              </span>
              <span className="text-[10px] font-semibold text-emerald-400">
                Citizen Civic Reporting Portal
              </span>
            </div>
          </div>

          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              type="button"
              onClick={() => handleTabChange('report')}
              className={`rounded-lg px-3 py-1.5 text-xs font-bold transition sm:text-sm ${
                activeTab === 'report'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              Report Waste
            </button>
            <button
              type="button"
              onClick={() => handleTabChange('reports')}
              className={`rounded-lg px-3 py-1.5 text-xs font-bold transition sm:text-sm ${
                activeTab === 'reports'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              My Reports ({myIncidents.length})
            </button>
            <button
              type="button"
              onClick={() => handleTabChange('profile')}
              className={`rounded-lg px-3 py-1.5 text-xs font-bold transition sm:text-sm ${
                activeTab === 'profile'
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-300 hover:bg-slate-800 hover:text-white'
              }`}
            >
              Profile
            </button>
            <button
              type="button"
              onClick={onLogout}
              className="ml-1 inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800 px-3 py-1.5 text-xs font-bold text-slate-200 hover:border-red-600 hover:bg-red-600 hover:text-white sm:text-sm"
            >
              <LogOut className="h-3.5 w-3.5" />
              <span>Logout</span>
            </button>
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-8 sm:px-6">
        {/* TAB 1: REPORT WASTE */}
        {activeTab === 'report' && (
          <div className="space-y-6">
            {/* Hero Section + 4-Step Progress Bar */}
            <div className="rounded-2xl bg-[#0F172A] p-6 text-white shadow-md sm:p-8">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h1 className="text-2xl font-extrabold sm:text-3xl">
                    Report Civic Waste
                  </h1>
                  <p className="mt-1.5 max-w-2xl text-sm text-slate-300">
                    Help your city stay clean. Upload a photograph and CleanPath AI
                    will automatically detect waste categories, calculate severity,
                    and geolocate the hotspot for municipal crew dispatch.
                  </p>
                </div>
              </div>

              {/* 4-Step Progress Indicator */}
              <div className="mt-6 grid grid-cols-2 gap-2.5 border-t border-slate-800 pt-5 sm:grid-cols-4">
                {workflowSteps.map((step, idx) => {
                  const isCurrent = currentStep === idx + 1;
                  return (
                    <div
                      key={step.num}
                      className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-2.5 transition ${
                        isCurrent
                          ? 'border-emerald-500 bg-emerald-600/20 text-white'
                          : step.active
                          ? 'border-emerald-800/80 bg-slate-900/90 text-emerald-400'
                          : 'border-slate-800 bg-slate-900/40 text-slate-500'
                      }`}
                    >
                      <span
                        className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-xs font-black ${
                          step.active
                            ? 'bg-emerald-600 text-white'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {step.num}
                      </span>
                      <span className="text-xs font-bold">{step.label}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {error && (
              <div className="flex items-center gap-2.5 rounded-xl border border-red-200 bg-red-50 p-4 text-sm font-semibold text-red-700">
                <AlertTriangle className="h-5 w-5 shrink-0 text-red-600" />
                <span>{error}</span>
              </div>
            )}

            {/* Submission Confirmation Banner */}
            {submittedIncident && (
              <div className="rounded-2xl border-2 border-emerald-500 bg-emerald-50 p-6 shadow-sm">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="mt-0.5 h-7 w-7 shrink-0 text-emerald-600" />
                    <div>
                      <h2 className="text-lg font-extrabold text-emerald-950">
                        Report Submitted Successfully —{' '}
                        {submittedIncident.incident_uuid}
                      </h2>
                      <p className="mt-1 text-xs text-emerald-800">
                        Your report has been persisted into the municipal database at{' '}
                        <strong>{submittedIncident.address}</strong> and clustered into
                        50m Hotspot{' '}
                        <strong>
                          {submittedIncident.hotspot_code ||
                            `#${submittedIncident.hotspot_id}`}
                        </strong>{' '}
                        ({submittedIncident.duplicate_status || 'PRIMARY'}) for
                        sanitation crew dispatch.
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setSelectedReportModal(submittedIncident)}
                      className="inline-flex items-center gap-1.5 rounded-lg border border-emerald-600 bg-white px-3.5 py-2 text-xs font-bold text-emerald-700 hover:bg-emerald-100"
                    >
                      <Eye className="h-3.5 w-3.5" />
                      Inspect Report
                    </button>
                    <button
                      type="button"
                      onClick={() => handleTabChange('reports')}
                      className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white hover:bg-emerald-700"
                    >
                      Track Complaint Status
                    </button>
                    <button
                      type="button"
                      onClick={handleResetReport}
                      className="rounded-lg border border-slate-300 bg-white px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-100"
                    >
                      Report Another
                    </button>
                  </div>
                </div>
              </div>
            )}

            {/* Upload Box when no image is currently analyzed */}
            {!analysis && (
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                className={`flex flex-col items-center justify-center rounded-2xl border-2 border-dashed p-10 text-center transition ${
                  dragActive
                    ? 'border-emerald-600 bg-emerald-50/60'
                    : 'border-slate-300 bg-white hover:border-emerald-500'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                  onChange={handleFileChange}
                  className="hidden"
                />

                <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-2xl bg-emerald-50 text-emerald-600">
                  {analyzing ? (
                    <Loader2 className="h-8 w-8 animate-spin" />
                  ) : (
                    <Upload className="h-8 w-8" />
                  )}
                </div>

                <h3 className="text-lg font-extrabold text-slate-900">
                  {analyzing
                    ? ANALYSIS_PROGRESS_STEPS[analysisStepIdx]
                    : 'Upload Waste Photograph'}
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  {analyzing
                    ? 'Executing multi-stage computer vision and EXIF geolocation pipeline...'
                    : 'Drag & Drop a .JPG or .PNG image here'}
                </p>

                {analyzing && (
                  <div className="mt-4 flex flex-wrap justify-center gap-2">
                    {ANALYSIS_PROGRESS_STEPS.map((label, idx) => (
                      <span
                        key={label}
                        className={`rounded-full px-3 py-1 text-[11px] font-semibold ${
                          idx <= analysisStepIdx
                            ? 'bg-emerald-100 text-emerald-800'
                            : 'bg-slate-100 text-slate-400'
                        }`}
                      >
                        {idx < analysisStepIdx ? '✓ ' : ''}
                        {label.replace('...', '')}
                      </span>
                    ))}
                  </div>
                )}

                <div className="my-4 flex items-center gap-3 text-xs font-bold uppercase text-slate-400">
                  <span className="h-px w-12 bg-slate-200" />
                  <span>OR</span>
                  <span className="h-px w-12 bg-slate-200" />
                </div>

                <div className="flex flex-wrap items-center justify-center gap-3">
                  <button
                    type="button"
                    disabled={analyzing}
                    onClick={() => fileInputRef.current?.click()}
                    className="inline-flex items-center gap-2 rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-bold text-white shadow-xs hover:bg-emerald-700 disabled:opacity-50"
                  >
                    <Upload className="h-4 w-4" />
                    <span>Upload Image</span>
                  </button>

                  <button
                    type="button"
                    disabled={analyzing}
                    onClick={handleOpenCamera}
                    className="inline-flex items-center gap-2 rounded-xl border border-slate-300 bg-slate-50 px-5 py-2.5 text-sm font-bold text-slate-800 hover:bg-slate-100 disabled:opacity-50"
                  >
                    <Camera className="h-4 w-4 text-emerald-600" />
                    <span>Open Camera</span>
                  </button>
                </div>
              </div>
            )}

            {/* Camera Capture Modal */}
            {cameraOpen && (
              <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/75 p-4">
                <div className="w-full max-w-lg overflow-hidden rounded-2xl bg-white shadow-2xl">
                  <div className="flex items-center justify-between bg-[#0F172A] px-5 py-3.5 text-white">
                    <span className="text-sm font-bold">Capture Waste Photograph</span>
                    <button
                      type="button"
                      onClick={stopCameraStream}
                      className="rounded p-1 text-slate-300 hover:text-white"
                    >
                      <X className="h-5 w-5" />
                    </button>
                  </div>
                  <div className="p-4">
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      className="h-72 w-full rounded-lg bg-black object-cover"
                    />
                    <div className="mt-4 flex justify-end gap-3">
                      <button
                        type="button"
                        onClick={stopCameraStream}
                        className="rounded-lg border border-slate-300 px-4 py-2 text-xs font-bold text-slate-700"
                      >
                        Cancel
                      </button>
                      <button
                        type="button"
                        onClick={handleCapturePhoto}
                        className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-600 px-5 py-2 text-xs font-bold text-white hover:bg-emerald-700"
                      >
                        <Camera className="h-4 w-4" />
                        Capture Photo
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Citizen Analysis Screen */}
            {analysis && (
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
                {/* Left Column: Image Preview with Bounding Boxes */}
                <div className="space-y-4 lg:col-span-7">
                  <div className="flex items-center justify-between">
                    <h2 className="text-base font-extrabold text-slate-900">
                      Image Preview &amp; AI Bounding Box Overlay
                    </h2>
                    <button
                      type="button"
                      onClick={handleResetReport}
                      className="inline-flex items-center gap-1 text-xs font-bold text-slate-600 hover:text-slate-900"
                    >
                      <RefreshCw className="h-3.5 w-3.5" />
                      Change Image
                    </button>
                  </div>

                  <DetectionOverlay
                    imageUrl={resolveImageUrl(analysis.image_url)}
                    imageWidth={analysis.image_width}
                    imageHeight={analysis.image_height}
                    detections={analysis.detections}
                    inferenceMode={analysis.inference_mode}
                  />

                  {/* Optional Citizen Notes / Landmark Description */}
                  <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                    <label
                      htmlFor="citizen-waste-notes"
                      className="block text-xs font-bold uppercase tracking-wider text-slate-600"
                    >
                      Landmark / Additional Notes (Optional)
                    </label>
                    <input
                      id="citizen-waste-notes"
                      type="text"
                      value={description}
                      onChange={(e) => setDescription(e.target.value)}
                      placeholder="e.g. Near Metro Gate 2, overflowing roadside bin blocking footpath..."
                      className="mt-1.5 w-full rounded-lg border border-slate-200 bg-slate-50 px-3.5 py-2 text-sm text-slate-900 focus:border-emerald-600 focus:bg-white focus:outline-none"
                    />
                  </div>
                </div>

                {/* Right Column: AI Detection, Severity, Location, Submit */}
                <div className="space-y-5 lg:col-span-5">
                  <SeverityCard
                    severityScore={analysis.severity_score}
                    severityTier={analysis.severity_tier}
                    categoryCounts={analysis.category_counts}
                    categoryStats={analysis.category_stats}
                    detections={analysis.detections}
                    inferenceMode={analysis.inference_mode}
                  />

                  {/* Location Status Box */}
                  <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
                    <h3 className="mb-3 flex items-center gap-2 text-base font-bold text-slate-900">
                      <MapPin className="h-5 w-5 text-emerald-600" />
                      Location Status
                    </h3>

                    {analysis.gps.source === 'EXIF' && !showManualMapOverride ? (
                      <div className="space-y-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3.5 text-xs">
                        <div className="flex items-center gap-1.5 font-bold text-emerald-800">
                          <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                          <span>GPS extracted from image EXIF metadata</span>
                        </div>
                        {address && (
                          <p className="font-semibold text-slate-800">{address}</p>
                        )}
                        <div className="grid grid-cols-2 gap-2 pt-1 text-slate-700">
                          <div>
                            <span className="text-slate-500">Latitude:</span>{' '}
                            <strong className="text-slate-900">
                              {latitude?.toFixed(5)}
                            </strong>
                          </div>
                          <div>
                            <span className="text-slate-500">Longitude:</span>{' '}
                            <strong className="text-slate-900">
                              {longitude?.toFixed(5)}
                            </strong>
                          </div>
                        </div>
                        <button
                          type="button"
                          onClick={() => setShowManualMapOverride(true)}
                          className="mt-1 text-[11px] font-bold text-emerald-700 underline"
                        >
                          Open interactive map to verify or adjust pin
                        </button>
                      </div>
                    ) : (
                      <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900">
                        <div className="flex items-center gap-1.5 font-bold">
                          <AlertTriangle className="h-4 w-4 text-amber-600" />
                          <span>
                            {analysis.gps.source === 'EXIF'
                              ? 'Manual / Interactive Map Mode Active'
                              : 'EXIF GPS not found in image'}
                          </span>
                        </div>
                        <p className="mt-1 text-amber-800">
                          Confirm or adjust the waste coordinates below using Current
                          Location or by dragging the red pin on the map.
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Submit Complaint Button */}
                  {!submittedIncident && (
                    <button
                      type="button"
                      disabled={
                        submitting || latitude === null || longitude === null
                      }
                      onClick={handleSubmitComplaint}
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 py-3.5 text-sm font-extrabold text-white shadow-md transition hover:bg-emerald-700 disabled:opacity-50"
                    >
                      <Send className="h-4 w-4" />
                      <span>
                        {submitting ? 'Submitting Complaint...' : 'Submit Complaint'}
                      </span>
                    </button>
                  )}
                </div>

                {/* Full-width Interactive Leaflet Map Pin Picker */}
                {(analysis.gps.source === 'MANUAL' || showManualMapOverride) && (
                  <div className="lg:col-span-12">
                    <MapPicker
                      latitude={latitude}
                      longitude={longitude}
                      locationSource={locationSource}
                      exifCoords={{
                        latitude: analysis.gps.latitude,
                        longitude: analysis.gps.longitude,
                      }}
                      address={address}
                      onChangeCoordinates={handleCoordinateChange}
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* TAB 2: MY REPORTS (INCIDENT TRACKING) */}
        {activeTab === 'reports' && (
          <div className="space-y-5">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <h2 className="text-2xl font-extrabold text-slate-900">
                  My Submitted Reports
                </h2>
                <p className="text-xs text-slate-500">
                  Real-time municipal dispatch, hotspot clustering, and resolution
                  tracking for your submitted civic waste reports.
                </p>
              </div>
              <button
                type="button"
                onClick={loadMyIncidents}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
              >
                <RefreshCw className="h-3.5 w-3.5" />
                Refresh Status
              </button>
            </div>

            {loadingIncidents ? (
              <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">
                Loading your submitted complaints...
              </div>
            ) : myIncidents.length === 0 ? (
              <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center">
                <FileText className="mx-auto mb-3 h-10 w-10 text-slate-400" />
                <h3 className="text-base font-bold text-slate-800">
                  No citizen reports yet.
                </h3>
                <p className="mt-1 text-xs text-slate-500">
                  Upload a waste photograph to generate an AI severity report and
                  track cleanup progress.
                </p>
                <button
                  type="button"
                  onClick={() => handleTabChange('report')}
                  className="mt-4 rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white hover:bg-emerald-700"
                >
                  Report Waste Now
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {myIncidents.map((inc) => (
                  <IncidentCard
                    key={inc.id}
                    incident={inc}
                    onSelect={(report) => setSelectedReportModal(report)}
                  />
                ))}
              </div>
            )}
          </div>
        )}

        {/* TAB 3: CITIZEN PROFILE */}
        {activeTab === 'profile' && (
          <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm">
            <div className="flex items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-100 text-emerald-700">
                <UserIcon className="h-7 w-7" />
              </div>
              <div>
                <h2 className="text-lg font-extrabold text-slate-900">
                  {user ? user.name || user.username : 'Guest Citizen'}
                </h2>
                <p className="text-xs text-slate-500">
                  {user ? user.email : 'Anonymous Local Session'}
                </p>
                <span className="mt-1 inline-block rounded-full bg-emerald-100 px-2.5 py-0.5 text-[11px] font-bold uppercase text-emerald-800">
                  {isGuest ? 'Guest Mode' : user?.role || 'citizen'}
                </span>
              </div>
            </div>

            <div className="mt-6 grid grid-cols-1 gap-4 border-t border-slate-100 pt-5 sm:grid-cols-3">
              <div className="rounded-xl bg-slate-50 p-4">
                <span className="text-xs text-slate-500">Submitted Complaints</span>
                <p className="mt-1 text-2xl font-black text-slate-900">
                  {myIncidents.length}
                </p>
              </div>
              <div className="rounded-xl bg-slate-50 p-4">
                <span className="text-xs text-slate-500">Dispatched / Active</span>
                <p className="mt-1 text-2xl font-black text-emerald-600">
                  {
                    myIncidents.filter((i) =>
                      ['ACKNOWLEDGED', 'DISPATCHED', 'IN_PROGRESS'].includes(i.status)
                    ).length
                  }
                </p>
              </div>
              <div className="rounded-xl bg-slate-50 p-4">
                <span className="text-xs text-slate-500">Resolved</span>
                <p className="mt-1 text-2xl font-black text-slate-900">
                  {myIncidents.filter((i) => i.status === 'RESOLVED').length}
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Full Report Detail Modal */}
      <ReportDetailModal
        report={selectedReportModal}
        onClose={() => setSelectedReportModal(null)}
        isOfficer={false}
      />
    </div>
  );
};
