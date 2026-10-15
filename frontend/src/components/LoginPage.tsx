import React, { useEffect, useState } from 'react';
import axios from 'axios';
import {
  AlertCircle,
  ArrowRight,
  Leaf,
  Lock,
  RefreshCw,
  Settings,
  ShieldCheck,
  User as UserIcon,
  UserCheck,
} from 'lucide-react';
import { getApiBaseUrl, loginApi, registerApi, testBackendHealth } from '../api';
import type { UserProfile, UserRole } from '../types';
import { ServerConfigModal } from './ServerConfigModal';

interface LoginPageProps {
  onAuthenticated: (token: string, user: UserProfile) => void;
  onContinueAsGuest: () => void;
}

export const LoginPage: React.FC<LoginPageProps> = ({
  onAuthenticated,
  onContinueAsGuest,
}) => {
  const [selectedRole, setSelectedRole] = useState<UserRole>('citizen');
  const [isRegisterMode, setIsRegisterMode] = useState<boolean>(false);
  const [username, setUsername] = useState<string>('citizen');
  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>('citizen123');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [isNetworkError, setIsNetworkError] = useState<boolean>(false);
  const [isServerModalOpen, setIsServerModalOpen] = useState<boolean>(false);
  const [currentApiUrl, setCurrentApiUrl] = useState<string>(() => getApiBaseUrl());
  const [isPingingServer, setIsPingingServer] = useState<boolean>(false);

  useEffect(() => {
    const handleUrlChange = (e: Event) => {
      const customEvent = e as CustomEvent<string>;
      setCurrentApiUrl(customEvent.detail || getApiBaseUrl());
    };
    window.addEventListener('terrasight_api_url_changed', handleUrlChange);
    return () => {
      window.removeEventListener('terrasight_api_url_changed', handleUrlChange);
    };
  }, []);

  const handleRoleSwitch = (role: UserRole) => {
    setSelectedRole(role);
    setIsRegisterMode(false);
    setError(null);
    setIsNetworkError(false);
    if (role === 'citizen') {
      setUsername('citizen');
      setPassword('citizen123');
    } else {
      setUsername('officer');
      setPassword('officer123');
    }
  };

  const handleWakeServer = async () => {
    setIsPingingServer(true);
    try {
      const res = await testBackendHealth();
      if (res.ok) {
        setError(null);
        setIsNetworkError(false);
      } else {
        setError(res.message);
      }
    } finally {
      setIsPingingServer(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsNetworkError(false);
    setLoading(true);

    try {
      if (isRegisterMode && selectedRole === 'citizen') {
        const res = await registerApi(
          username.trim(),
          email.trim() || `${username.trim().toLowerCase()}@cleanpath.local`,
          password,
          'citizen'
        );
        onAuthenticated(res.access_token, res.user);
      } else {
        const res = await loginApi(username.trim(), password);
        if (selectedRole === 'officer' && res.user.role !== 'officer') {
          setError('Unauthorized: This account does not have Municipal Officer privileges.');
          setLoading(false);
          return;
        }
        onAuthenticated(res.access_token, res.user);
      }
    } catch (err: unknown) {
      let detail = 'Authentication failed. Please check your credentials.';
      let netErr = false;

      if (axios.isAxiosError(err)) {
        if (!err.response) {
          netErr = true;
          if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
            detail = `Connection timed out reaching backend (${getApiBaseUrl()}). If your Render service was sleeping, it takes 30-50s to spin up. Please retry in a few moments.`;
          } else {
            detail = `Unable to connect to backend server at ${getApiBaseUrl()}. Please verify your backend is active or update the server URL below.`;
          }
        } else if (err.response.status === 404) {
          netErr = true;
          detail = `API endpoint not found (404) at ${getApiBaseUrl()}. Please verify that the Backend Server URL points to the FastAPI service.`;
        } else if (err.response.status === 401) {
          detail = 'Invalid credentials. Please verify your username and password.';
        } else if (
          err.response.data &&
          typeof err.response.data === 'object' &&
          'detail' in err.response.data &&
          typeof (err.response.data as { detail: string }).detail === 'string'
        ) {
          detail = (err.response.data as { detail: string }).detail;
        }
      } else if (err instanceof Error) {
        detail = err.message;
      }

      setError(detail);
      setIsNetworkError(netErr);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen flex-col bg-[#F8FAFC]">
      {/* Dark Navy Header */}
      <header className="bg-[#0F172A] px-6 py-4 text-white shadow-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-white shadow-xs">
              <Leaf className="h-6 w-6" />
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tight text-white">
                CLEANPATH AI
              </h1>
              <p className="text-xs font-medium text-emerald-400">
                TerraSight AI — Detect. Locate. Cluster. Prioritize. Dispatch.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setIsServerModalOpen(true)}
              className="flex items-center gap-2 rounded-full border border-slate-700 bg-slate-800/90 px-3.5 py-1.5 text-xs font-semibold text-slate-200 transition hover:border-slate-500 hover:bg-slate-700"
              title="Click to configure Backend API endpoint"
            >
              <span className="h-2 w-2 rounded-full bg-emerald-400"></span>
              <span className="max-w-[180px] sm:max-w-[260px] truncate font-mono text-[11px] text-slate-300">
                {currentApiUrl}
              </span>
              <Settings className="h-3.5 w-3.5 text-slate-400" />
            </button>
            <span className="hidden rounded-full border border-slate-700 bg-slate-800/80 px-3 py-1 text-xs font-semibold text-slate-300 md:inline-block">
              100% Local Civic-Tech
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto flex w-full max-w-5xl flex-1 items-center justify-center px-4 py-10">
        <div className="grid w-full grid-cols-1 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-xl lg:grid-cols-12">
          {/* Left Side Branding */}
          <div className="flex flex-col justify-between bg-[#0F172A] p-8 text-white lg:col-span-5">
            <div className="space-y-6">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-500/20 px-3 py-1 text-xs font-bold text-emerald-400">
                AI &amp; Geospatial Waste Intelligence
              </span>
              <h2 className="text-2xl font-extrabold leading-tight sm:text-3xl">
                Smart Civic Waste Reporting &amp; Municipal Dispatch
              </h2>
              <p className="text-sm leading-relaxed text-slate-300">
                Upload waste photographs for automated YOLOv8 category detection,
                EXIF GPS localization, mathematical severity scoring, and 50-meter
                Haversine DBSCAN hotspot clustering.
              </p>
            </div>

            <div className="mt-8 space-y-3 border-t border-slate-800 pt-6 text-xs text-slate-300">
              <div className="font-bold uppercase tracking-wider text-emerald-400">
                Demo Credentials
              </div>
              <div className="flex items-center justify-between rounded-lg bg-slate-800/90 px-3 py-2">
                <span>Citizen Portal:</span>
                <code className="font-mono font-bold text-white">
                  citizen / citizen123
                </code>
              </div>
              <div className="flex items-center justify-between rounded-lg bg-slate-800/90 px-3 py-2">
                <span>Officer Command:</span>
                <code className="font-mono font-bold text-white">
                  officer / officer123
                </code>
              </div>
            </div>
          </div>

          {/* Right Side Login Form */}
          <div className="p-8 lg:col-span-7">
            {/* Role Selector */}
            <div className="mb-6 grid grid-cols-2 gap-2 rounded-xl bg-slate-100 p-1.5">
              <button
                type="button"
                onClick={() => handleRoleSwitch('citizen')}
                className={`flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-bold transition ${
                  selectedRole === 'citizen'
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <UserIcon className="h-4 w-4 text-emerald-600" />
                Citizen
              </button>
              <button
                type="button"
                onClick={() => handleRoleSwitch('officer')}
                className={`flex items-center justify-center gap-2 rounded-lg py-2.5 text-sm font-bold transition ${
                  selectedRole === 'officer'
                    ? 'bg-[#0F172A] text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <ShieldCheck className="h-4 w-4 text-emerald-500" />
                Officer
              </button>
            </div>

            <div className="mb-6">
              <h3 className="text-xl font-extrabold text-slate-900">
                {selectedRole === 'citizen'
                  ? isRegisterMode
                    ? 'Create Citizen Account'
                    : 'Citizen Portal Login'
                  : 'Municipal Officer Command Login'}
              </h3>
              <p className="mt-1 text-xs text-slate-500">
                {selectedRole === 'citizen'
                  ? 'Sign in to report civic waste and track complaint resolution.'
                  : 'Restricted access for authorized municipal sanitation officers.'}
              </p>
            </div>

            {error && (
              isNetworkError ? (
                <div className="mb-5 rounded-lg border border-red-200 bg-red-50 p-3.5 text-xs text-red-800">
                  <div className="flex items-start gap-2.5">
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
                    <div className="flex-1 space-y-2">
                      <p className="font-semibold leading-relaxed">{error}</p>
                      <div className="flex flex-wrap items-center gap-2 pt-1">
                        <button
                          type="button"
                          onClick={() => setIsServerModalOpen(true)}
                          className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2.5 py-1 text-xs font-bold text-slate-800 shadow-xs transition hover:bg-slate-50"
                        >
                          <Settings className="h-3 w-3 text-slate-600" />
                          <span>Configure Server URL</span>
                        </button>
                        <button
                          type="button"
                          onClick={handleWakeServer}
                          disabled={isPingingServer}
                          className="flex items-center gap-1 rounded-md bg-red-600 px-2.5 py-1 text-xs font-bold text-white shadow-xs transition hover:bg-red-700 disabled:opacity-50"
                        >
                          <RefreshCw className={`h-3 w-3 ${isPingingServer ? 'animate-spin' : ''}`} />
                          <span>{isPingingServer ? 'Waking...' : 'Wake / Test Server'}</span>
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="mb-5 flex items-center gap-2 rounded-lg border border-red-200 bg-red-50 px-3.5 py-2.5 text-xs font-semibold text-red-700">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  <span>{error}</span>
                </div>
              )
            )}

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
                  {selectedRole === 'citizen' ? 'Email / Username' : 'Officer ID'}
                </label>
                <div className="relative">
                  <UserIcon className=" absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                  <input
                    type="text"
                    required
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder={
                      selectedRole === 'citizen'
                        ? 'Enter username or email'
                        : 'Enter Officer ID (e.g. officer)'
                    }
                    className="w-full rounded-lg border border-slate-300 py-2.5 pl-10 pr-4 text-sm text-slate-900 focus:border-emerald-600 focus:outline-none"
                  />
                </div>
              </div>

              {isRegisterMode && selectedRole === 'citizen' && (
                <div>
                  <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
                    Email Address
                  </label>
                  <input
                    type="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="citizen@example.com"
                    className="w-full rounded-lg border border-slate-300 px-3.5 py-2.5 text-sm text-slate-900 focus:border-emerald-600 focus:outline-none"
                  />
                </div>
              )}

              <div>
                <label className="mb-1.5 block text-xs font-bold uppercase tracking-wide text-slate-700">
                  Password
                </label>
                <div className="relative">
                  <Lock className="absolute left-3.5 top-3 h-4 w-4 text-slate-400" />
                  <input
                    type="password"
                    required
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter password"
                    className="w-full rounded-lg border border-slate-300 py-2.5 pl-10 pr-4 text-sm text-slate-900 focus:border-emerald-600 focus:outline-none"
                  />
                </div>
              </div>

              <button
                type="submit"
                disabled={loading}
                className={`mt-2 flex w-full items-center justify-center gap-2 rounded-lg py-3 text-sm font-bold text-white shadow-sm transition ${
                  selectedRole === 'citizen'
                    ? 'bg-emerald-600 hover:bg-emerald-700'
                    : 'bg-[#0F172A] hover:bg-slate-800'
                } disabled:opacity-50`}
              >
                {selectedRole === 'citizen' ? (
                  <>
                    <span>{loading ? 'Signing in...' : isRegisterMode ? 'Register & Login' : 'Login'}</span>
                    <ArrowRight className="h-4 w-4" />
                  </>
                ) : (
                  <>
                    <ShieldCheck className="h-4 w-4 text-emerald-400" />
                    <span>{loading ? 'Verifying...' : 'Secure Officer Login'}</span>
                  </>
                )}
              </button>

              {loading && (
                <p className="mt-2 text-center text-xs text-slate-500 animate-pulse">
                  Connecting to backend... {currentApiUrl.includes('onrender.com') ? '(Render free tier services may take 30-45s to wake up)' : ''}
                </p>
              )}
            </form>

            {selectedRole === 'citizen' && (
              <div className="mt-5 space-y-3 border-t border-slate-100 pt-5">
                <button
                  type="button"
                  onClick={onContinueAsGuest}
                  className="flex w-full items-center justify-center gap-2 rounded-lg border border-slate-300 bg-slate-50 py-2.5 text-sm font-bold text-slate-700 transition hover:bg-slate-100"
                >
                  <UserCheck className="h-4 w-4 text-emerald-600" />
                  Continue as Guest
                </button>

                <div className="text-center">
                  <button
                    type="button"
                    onClick={() => {
                      setIsRegisterMode(!isRegisterMode);
                      setError(null);
                      setIsNetworkError(false);
                    }}
                    className="text-xs font-semibold text-emerald-700 hover:underline"
                  >
                    {isRegisterMode
                      ? 'Already have an account? Sign in'
                      : 'Need a new citizen account? Register here'}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      <ServerConfigModal
        isOpen={isServerModalOpen}
        onClose={() => setIsServerModalOpen(false)}
        onSaved={(url) => {
          setCurrentApiUrl(url);
          setError(null);
          setIsNetworkError(false);
        }}
      />
    </div>
  );
};
