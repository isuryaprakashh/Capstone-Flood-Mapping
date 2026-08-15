import React, { useState, useRef, useCallback } from 'react';
import { MapContainer, TileLayer, ImageOverlay, Polyline, useMapEvents } from 'react-leaflet';
import axios from 'axios';
import 'leaflet/dist/leaflet.css';

/* ─── Constants ──────────────────────────────────────────────────── */
const API_BASE = '/api';
const CLASS_NAMES = ['building', 'road', 'flood', 'flooded_building', 'flooded_road'];
const CLASS_COLORS = {
  building: '#4285f4',
  road: '#dadce0',
  flood: '#00c8ff',
  flooded_building: '#ea4335',
  flooded_road: '#fbbc04',
};

/* ─── App ────────────────────────────────────────────────────────── */
export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [routePoints, setRoutePoints] = useState([]);
  const [routeResult, setRouteResult] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [apiStatus, setApiStatus] = useState(null);

  const mapRef = useRef(null);

  /* ─ Health Check ─ */
  const checkHealth = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setApiStatus(res.data);
    } catch {
      setApiStatus({ status: 'error' });
    }
  }, []);

  React.useEffect(() => { checkHealth(); }, [checkHealth]);

  /* ─ Predict ─ */
  const handlePredict = async () => {
    if (!preFile || !postFile) {
      setError('Please upload both pre-event and post-event images.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const form = new FormData();
      form.append('pre_image', preFile);
      form.append('post_image', postFile);

      const res = await axios.post(`${API_BASE}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (res.data.ok) {
        setResult(res.data);
        if (res.data.mask_png_base64) {
          setOverlayUrl(`data:image/png;base64,${res.data.mask_png_base64}`);
        }
        setActiveTab('results');
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Check API server.');
    } finally {
      setLoading(false);
    }
  };

  /* ─ Sample Demo ─ */
  const handleSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API_BASE}/sample`);
      if (res.data.ok) {
        setResult(res.data);
        if (res.data.mask_png_base64) {
          setOverlayUrl(`data:image/png;base64,${res.data.mask_png_base64}`);
        }
        setActiveTab('results');
      }
    } catch (err) {
      setError('Sample prediction failed.');
    } finally {
      setLoading(false);
    }
  };

  /* ─ Route ─ */
  const handleRoute = async () => {
    if (routePoints.length < 2) {
      setError('Click two points on the map to set start and end.');
      return;
    }
    setLoading(true);
    try {
      const [start, end] = routePoints;
      const form = new FormData();
      form.append('start_y', Math.round(start[0]));
      form.append('start_x', Math.round(start[1]));
      form.append('end_y', Math.round(end[0]));
      form.append('end_x', Math.round(end[1]));

      const res = await axios.post(`${API_BASE}/route`, form);
      if (res.data.ok) {
        setRouteResult(res.data);
      }
    } catch (err) {
      setError('Routing failed.');
    } finally {
      setLoading(false);
    }
  };

  const themeClass = darkMode ? 'dark' : '';

  return (
    <div className={`app-root ${themeClass}`}>
      {/* ─ Navbar ─ */}
      <nav className="navbar">
        <div className="nav-brand">
          <div className="brand-icon">SN8</div>
          <div className="brand-text">
            <span className="brand-title">Flood Mapping</span>
            <span className="brand-subtitle">SpaceNet 8 · Deep Learning</span>
          </div>
        </div>

        <div className="nav-tabs">
          {['upload', 'results', 'route'].map(tab => (
            <button
              key={tab}
              className={`nav-tab ${activeTab === tab ? 'active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        <div className="nav-actions">
          <div className={`status-dot ${apiStatus?.status === 'ok' ? 'online' : 'offline'}`} />
          <button className="theme-btn" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? '☀' : '🌙'}
          </button>
        </div>
      </nav>

      {/* ─ Main Content ─ */}
      <div className="main-layout">
        {/* Sidebar */}
        <aside className="sidebar">
          {error && (
            <div className="alert alert-error">
              <span>⚠</span> {error}
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}

          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">📡</span>
                Upload Satellite Images
              </h2>
              <p className="panel-desc">
                Upload pre-event and post-event GeoTIFF satellite image pairs
                for AI-powered flood damage analysis.
              </p>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-badge pre">PRE</span>
                  Pre-Event Image
                </label>
                <div className="upload-dropzone" onClick={() => document.getElementById('pre-input').click()}>
                  <input id="pre-input" type="file" accept=".tif,.tiff" hidden onChange={e => setPreFile(e.target.files[0])} />
                  {preFile ? (
                    <span className="upload-filename">✓ {preFile.name}</span>
                  ) : (
                    <span className="upload-placeholder">Click to select .tif file</span>
                  )}
                </div>
              </div>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-badge post">POST</span>
                  Post-Event Image
                </label>
                <div className="upload-dropzone" onClick={() => document.getElementById('post-input').click()}>
                  <input id="post-input" type="file" accept=".tif,.tiff" hidden onChange={e => setPostFile(e.target.files[0])} />
                  {postFile ? (
                    <span className="upload-filename">✓ {postFile.name}</span>
                  ) : (
                    <span className="upload-placeholder">Click to select .tif file</span>
                  )}
                </div>
              </div>

              <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                {loading ? '⟳ Processing...' : '🔍 Run Flood Analysis'}
              </button>

              <div className="divider">
                <span>or</span>
              </div>

              <button className="btn btn-outline" onClick={handleSample} disabled={loading}>
                📋 Load Sample Demo
              </button>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === 'results' && result && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">📊</span>
                Detection Results
              </h2>

              {result.is_stub && (
                <div className="alert alert-info">
                  <span>ℹ</span> Stub mode — demo data, no real model loaded.
                </div>
              )}

              <div className="results-grid">
                {result.class_names?.map(name => {
                  const stat = result.statistics?.[name];
                  if (!stat) return null;
                  return (
                    <div key={name} className="result-card" style={{ borderColor: CLASS_COLORS[name] }}>
                      <div className="result-header">
                        <div className="result-dot" style={{ background: CLASS_COLORS[name] }} />
                        <span className="result-name">{name.replace('_', ' ')}</span>
                      </div>
                      <div className="result-value">{stat.coverage_pct}%</div>
                      <div className="result-bar">
                        <div
                          className="result-bar-fill"
                          style={{
                            width: `${Math.min(stat.coverage_pct * 2, 100)}%`,
                            background: CLASS_COLORS[name],
                          }}
                        />
                      </div>
                      <div className="result-detail">
                        {stat.positive_pixels?.toLocaleString()} px detected
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Route Tab */}
          {activeTab === 'route' && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">🗺</span>
                Safe Route Finder
              </h2>
              <p className="panel-desc">
                Click two points on the map to set start and destination.
                The router will find the shortest path avoiding flooded roads.
              </p>

              <div className="route-points">
                <div className="route-point">
                  <span className="route-marker start">A</span>
                  <span>{routePoints[0] ? `(${routePoints[0][0]}, ${routePoints[0][1]})` : 'Click map...'}</span>
                </div>
                <div className="route-point">
                  <span className="route-marker end">B</span>
                  <span>{routePoints[1] ? `(${routePoints[1][0]}, ${routePoints[1][1]})` : 'Click map...'}</span>
                </div>
              </div>

              <button className="btn btn-primary" onClick={handleRoute} disabled={loading || routePoints.length < 2}>
                🚗 Find Safe Route
              </button>

              {routeResult && (
                <div className="route-result">
                  <div className={`route-status ${routeResult.route_type}`}>
                    {routeResult.route_type === 'safe' ? '✅ Safe Route Found' : '⚠ Route Uses Flooded Roads'}
                  </div>
                  <div className="route-distance">
                    Distance: {routeResult.distance_m}m
                  </div>
                </div>
              )}

              <button className="btn btn-outline" onClick={() => { setRoutePoints([]); setRouteResult(null); }}>
                🔄 Clear Route
              </button>
            </div>
          )}
        </aside>

        {/* Map */}
        <div className="map-container">
          <MapContainer
            center={[30.0, -90.0]}
            zoom={10}
            style={{ height: '100%', width: '100%' }}
            ref={mapRef}
          >
            <TileLayer
              url={darkMode
                ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png'
                : 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
              }
              attribution="&copy; Esri / CartoDB"
            />

            {overlayUrl && (
              <ImageOverlay
                url={overlayUrl}
                bounds={[[29.8, -90.3], [30.2, -89.7]]}
                opacity={0.7}
              />
            )}

            {routeResult?.route?.geometry?.coordinates?.length > 0 && (
              <Polyline
                positions={routeResult.route.geometry.coordinates.map(([x, y]) => [y, x])}
                color={routeResult.has_flooded_segments ? '#ea4335' : '#34a853'}
                weight={4}
                dashArray={routeResult.has_flooded_segments ? '10 5' : undefined}
              />
            )}

            <MapClickHandler
              routePoints={routePoints}
              setRoutePoints={setRoutePoints}
              activeTab={activeTab}
            />
          </MapContainer>

          {/* Legend */}
          <div className="map-legend">
            <div className="legend-title">Legend</div>
            {CLASS_NAMES.map(name => (
              <div key={name} className="legend-item">
                <div className="legend-swatch" style={{ background: CLASS_COLORS[name] }} />
                <span>{name.replace('_', ' ')}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ─── Map Click Handler ──────────────────────────────────────────── */
function MapClickHandler({ routePoints, setRoutePoints, activeTab }) {
  useMapEvents({
    click(e) {
      if (activeTab !== 'route') return;
      const { lat, lng } = e.latlng;
      const point = [Math.round(lat * 100) / 100, Math.round(lng * 100) / 100];

      if (routePoints.length >= 2) {
        setRoutePoints([point]);
      } else {
        setRoutePoints([...routePoints, point]);
      }
    },
  });
  return null;
}
