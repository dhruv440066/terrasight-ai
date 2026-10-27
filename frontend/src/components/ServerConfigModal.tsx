import React, { useState } from 'react';
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Globe,
  RefreshCw,
  Server,
  X,
} from 'lucide-react';
import {
  getApiBaseUrl,
  normalizeApiUrl,
  resetApiBaseUrl,
  setApiBaseUrl,
  testBackendHealth,
} from '../api';

interface ServerConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSaved?: (newUrl: string) => void;
}

export const ServerConfigModal: React.FC<ServerConfigModalProps> = ({
  isOpen,
  onClose,
  onSaved,
}) => {
  const [urlInput, setUrlInput] = useState<string>(() => getApiBaseUrl());
  const [testing, setTesting] = useState<boolean>(false);
  const [testResult, setTestResult] = useState<{
    ok: boolean;
    message: string;
    latencyMs?: number;
  } | null>(null);

  if (!isOpen) return null;

  const handleTest = async () => {
    const target = normalizeApiUrl(urlInput);
    if (!target) return;
    setTesting(true);
    setTestResult(null);
    try {
      const res = await testBackendHealth(target);
      setTestResult(res);
    } finally {
      setTesting(false);
    }
  };

  const handleSave = () => {
    const target = normalizeApiUrl(urlInput);
    if (!target) return;
    setApiBaseUrl(target);
    if (onSaved) onSaved(target);
    onClose();
  };

  const handleReset = () => {
    resetApiBaseUrl();
    const detected = getApiBaseUrl();
    setUrlInput(detected);
    setTestResult(null);
    if (onSaved) onSaved(detected);
    onClose();
  };

  const setPreset = (presetUrl: string) => {
    setUrlInput(presetUrl);
    setTestResult(null);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 p-4 backdrop-blur-xs">
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-6 shadow-2xl">
        <div className="flex items-center justify-between border-b border-slate-100 pb-4">
          <div className="flex items-center gap-2.5">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-900 text-emerald-400">
              <Server className="h-5 w-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-900">
                Backend Server Configuration
              </h3>
              <p className="text-xs text-slate-500">
                Configure the FastAPI + YOLOv8 + GIS backend API endpoint.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="mt-5 space-y-4">
          <div>
            <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
              Backend API URL
            </label>
            <div className="relative">
              <Globe className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
              <input
                type="text"
                value={urlInput}
                onChange={(e) => {
                  setUrlInput(e.target.value);
                  setTestResult(null);
                }}
                placeholder="https://terrasight-api.onrender.com"
                className="w-full rounded-lg border border-slate-300 py-2.5 pl-10 pr-4 text-sm font-medium text-slate-900 focus:border-emerald-600 focus:outline-none"
              />
            </div>
          </div>

          {/* Quick Presets */}
          <div>
            <span className="mb-1.5 block text-xs font-semibold text-slate-500">
              Quick Preset Targets:
            </span>
            <div className="flex flex-wrap gap-2 text-xs">
              {typeof window !== 'undefined' && (
                <button
                  type="button"
                  onClick={() => setPreset(window.location.origin)}
                  className="rounded-md border border-emerald-300 bg-emerald-50 px-2.5 py-1 font-semibold text-emerald-800 transition hover:bg-emerald-100"
                >
                  Current Origin ({window.location.host})
                </button>
              )}
              <button
                type="button"
                onClick={() => setPreset('https://terrasight-ai.onrender.com')}
                className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 font-medium text-slate-700 transition hover:bg-slate-100"
              >
                Render Backend (terrasight-ai)
              </button>
              <button
                type="button"
                onClick={() => setPreset('http://localhost:8000')}
                className="rounded-md border border-slate-200 bg-slate-50 px-2.5 py-1 font-medium text-slate-700 transition hover:bg-slate-100"
              >
                Localhost (8000)
              </button>
            </div>
          </div>

          {/* Test Connection Button & Status */}
          <div className="rounded-xl border border-slate-100 bg-slate-50 p-3.5">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-slate-700">
                Server Connectivity Status
              </span>
              <button
                type="button"
                onClick={handleTest}
                disabled={testing}
                className="flex items-center gap-1.5 rounded-lg bg-white px-3 py-1.5 text-xs font-bold text-slate-800 shadow-xs ring-1 ring-slate-200 transition hover:bg-slate-50 disabled:opacity-50"
              >
                <Activity
                  className={`h-3.5 w-3.5 text-emerald-600 ${
                    testing ? 'animate-spin' : ''
                  }`}
                />
                <span>{testing ? 'Pinging Server...' : 'Test Connection'}</span>
              </button>
            </div>

            {testResult && (
              <div
                className={`mt-3 flex items-start gap-2 rounded-lg p-2.5 text-xs font-medium ${
                  testResult.ok
                    ? 'border border-emerald-200 bg-emerald-50 text-emerald-800'
                    : 'border border-amber-200 bg-amber-50 text-amber-900'
                }`}
              >
                {testResult.ok ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
                ) : (
                  <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
                )}
                <div>
                  <p className="font-bold">{testResult.message}</p>
                  {!testResult.ok && (
                    <p className="mt-1 text-[11px] leading-relaxed text-amber-700">
                      If using Render free tier, inactive services take 30–50s to wake up.
                      Click &quot;Test Connection&quot; again in a few seconds once it has started.
                    </p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="mt-6 flex items-center justify-between border-t border-slate-100 pt-4">
          <button
            type="button"
            onClick={handleReset}
            className="flex items-center gap-1 text-xs font-semibold text-slate-500 hover:text-slate-800"
          >
            <RefreshCw className="h-3 w-3" />
            <span>Reset to Auto-Detect</span>
          </button>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-slate-300 px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50"
            >
              Cancel
            </button>
            <button
              type="button"
              onClick={handleSave}
              className="rounded-lg bg-emerald-600 px-4 py-2 text-xs font-bold text-white shadow-xs hover:bg-emerald-700"
            >
              Save &amp; Connect
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
