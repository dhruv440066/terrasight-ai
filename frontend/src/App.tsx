import React, { useEffect, useState } from 'react';
import {
  BrowserRouter,
  Navigate,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom';
import { fetchCurrentUser } from './api';
import { CitizenView } from './components/CitizenView';
import { LoginPage } from './components/LoginPage';
import { OfficerGIS } from './components/OfficerGIS';
import type { UserProfile } from './types';

const TOKEN_KEY = 'cleanpath_token';
const USER_KEY = 'cleanpath_user';
const GUEST_MODE_KEY = 'cleanpath_guest_mode';

const AppRoutes: React.FC = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState<UserProfile | null>(() => {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as UserProfile;
    } catch {
      return null;
    }
  });
  const [isGuest, setIsGuest] = useState<boolean>(() => {
    return localStorage.getItem(GUEST_MODE_KEY) === 'true';
  });
  const [checkingAuth, setCheckingAuth] = useState<boolean>(true);

  useEffect(() => {
    const token = localStorage.getItem(TOKEN_KEY);
    if (!token) {
      setCheckingAuth(false);
      return;
    }
    fetchCurrentUser()
      .then((profile) => {
        setUser(profile);
        localStorage.setItem(USER_KEY, JSON.stringify(profile));
      })
      .catch(() => {
        localStorage.removeItem(TOKEN_KEY);
        localStorage.removeItem(USER_KEY);
        setUser(null);
      })
      .finally(() => {
        setCheckingAuth(false);
      });
  }, []);

  const handleAuthenticated = (token: string, profile: UserProfile) => {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(USER_KEY, JSON.stringify(profile));
    localStorage.removeItem(GUEST_MODE_KEY);
    setUser(profile);
    setIsGuest(false);

    if (profile.role === 'officer') {
      navigate('/officer/map', { replace: true });
    } else {
      navigate('/citizen/report', { replace: true });
    }
  };

  const handleContinueAsGuest = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.setItem(GUEST_MODE_KEY, 'true');
    setUser(null);
    setIsGuest(true);
    navigate('/citizen/report', { replace: true });
  };

  const handleLogout = () => {
    localStorage.removeItem(TOKEN_KEY);
    localStorage.removeItem(USER_KEY);
    localStorage.removeItem(GUEST_MODE_KEY);
    setUser(null);
    setIsGuest(false);
    navigate('/login', { replace: true });
  };

  if (checkingAuth) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#0F172A] text-white">
        <div className="text-center">
          <div className="mx-auto mb-3 h-8 w-8 animate-spin rounded-full border-2 border-emerald-400 border-t-transparent" />
          <p className="text-sm font-bold">Initializing CleanPath AI...</p>
        </div>
      </div>
    );
  }

  const isCitizenAuthorized = Boolean(user) || isGuest;
  const isOfficerAuthorized = Boolean(user && user.role === 'officer');

  return (
    <Routes>
      <Route
        path="/login"
        element={
          <LoginPage
            onAuthenticated={handleAuthenticated}
            onContinueAsGuest={handleContinueAsGuest}
          />
        }
      />

      {/* Citizen Portal Routes */}
      <Route
        path="/citizen"
        element={
          isCitizenAuthorized ? (
            <CitizenView user={user} isGuest={isGuest} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/citizen/report"
        element={
          isCitizenAuthorized ? (
            <CitizenView user={user} isGuest={isGuest} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/citizen/reports"
        element={
          isCitizenAuthorized ? (
            <CitizenView user={user} isGuest={isGuest} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />

      {/* Officer Command Center Routes (RBAC Protected) */}
      <Route
        path="/officer"
        element={
          isOfficerAuthorized && user ? (
            <OfficerGIS user={user} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/officer/map"
        element={
          isOfficerAuthorized && user ? (
            <OfficerGIS user={user} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
      <Route
        path="/officer/analytics"
        element={
          isOfficerAuthorized && user ? (
            <OfficerGIS user={user} onLogout={handleLogout} />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />

      {/* Default Fallback Redirect */}
      <Route
        path="*"
        element={
          isOfficerAuthorized ? (
            <Navigate to="/officer/map" replace />
          ) : isCitizenAuthorized ? (
            <Navigate to="/citizen/report" replace />
          ) : (
            <Navigate to="/login" replace />
          )
        }
      />
    </Routes>
  );
};

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );
};

export default App;
