import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const CLASS_COLORS = {
  building: '#3b82f6',
  road: '#94a3b8',
  flood: '#06b6d4',
  flooded_building: '#ef4444',
  flooded_road: '#f59e0b',
};

export default function App() {
  const [darkMode, setDarkMode] = useState(true);
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [apiStatus, setApiStatus] = useState(null);

  // Visualization settings
  const [viewMode, setViewMode] = useState('hybrid'); // 'vectors' | 'raster' | 'hybrid'
  const [showRoads, setShowRoads] = useState(true);
  const [showBuildings, setShowBuildings] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.75);

  // Routing state
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [routeResult, setRouteResult] = useState(null);
  const [routingLoading, setRoutingLoading] = useState(false);
  const [routingMode, setRoutingMode] = useState('start'); // 'start' | 'end'

  // Canvas zoom & pan
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  /* ─── Health Check ─── */
  const checkHealth = useCallback(async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      setApiStatus(res.data);
    } catch {
      setApiStatus({ status: 'error' });
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  /* ─── Prediction Handler ─── */
  const handlePredict = async () => {
    if (!preFile || !postFile) {
      setError('Please upload both pre-event and post-event images.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    setRouteResult(null);
    setStartPoint(null);
    setEndPoint(null);

    try {
      const form = new FormData();
      form.append('pre_image', preFile);
      form.append('post_image', postFile);

      const res = await axios.post(`${API_BASE}/predict`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (res.data.ok) {
        setResult(res.data);
        setActiveTab('results');
        resetView();
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed. Check backend API.');
    } finally {
      setLoading(false);
    }
  };

  /* ─── Load Sample Demo ─── */
  const handleSample = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get(`${API_BASE}/sample`);
      if (res.data.ok) {
        setResult(res.data);
        setActiveTab('results');
        resetView();
      }
    } catch (err) {
      setError('Sample load failed.');
    } finally {
      setLoading(false);
    }
  };

  /* ─── Route Computation ─── */
  const calculateRoute = async (start, end) => {
    if (!start || !end) return;
    setRoutingLoading(true);
    try {
      const form = new FormData();
      form.append('start_y', Math.round(start.y));
      form.append('start_x', Math.round(start.x));
      form.append('end_y', Math.round(end.y));
      form.append('end_x', Math.round(end.x));

      const res = await axios.post(`${API_BASE}/route`, form);
      if (res.data.ok) {
        setRouteResult(res.data);
      }
    } catch (err) {
      setError('Routing calculation failed.');
    } finally {
      setRoutingLoading(false);
    }
  };

  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  /* ─── Canvas Render Engine ─── */
  const drawScene = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !result) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    // Save transform state
    ctx.save();
    ctx.translate(offset.x + width / 2, offset.y + height / 2);
    ctx.scale(scale, scale);
    ctx.translate(-width / 2, -height / 2);

    const maskShape = result.mask_shape || [5, 512, 512];
    const H = maskShape[1] || 512;
    const W = maskShape[2] || 512;

    // Draw background canvas
    ctx.fillStyle = darkMode ? '#0a0d14' : '#f8fafc';
    ctx.fillRect(0, 0, W, H);
    ctx.strokeStyle = darkMode ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, W, H);

    // 1. Draw Raster Mask Overlay if enabled
    if (showMask && (viewMode === 'raster' || viewMode === 'hybrid') && result.mask_png_base64) {
      const img = new Image();
      img.src = `data:image/png;base64,${result.mask_png_base64}`;
      if (img.complete) {
        ctx.globalAlpha = overlayOpacity;
        ctx.drawImage(img, 0, 0, W, H);
        ctx.globalAlpha = 1.0;
      } else {
        img.onload = () => drawScene();
      }
    }

    // 2. Draw Vector Buildings
    if (showBuildings && (viewMode === 'vectors' || viewMode === 'hybrid') && result.building_footprints?.features) {
      result.building_footprints.features.forEach((feature) => {
        const coords = feature.geometry?.coordinates?.[0];
        if (!coords || coords.length < 3) return;

        const isFlooded = feature.properties?.flooded;
        ctx.beginPath();
        ctx.moveTo(coords[0][0], coords[0][1]);
        for (let i = 1; i < coords.length; i++) {
          ctx.lineTo(coords[i][0], coords[i][1]);
        }
        ctx.closePath();

        ctx.fillStyle = isFlooded ? 'rgba(239, 68, 68, 0.4)' : 'rgba(59, 130, 246, 0.35)';
        ctx.fill();
        ctx.strokeStyle = isFlooded ? '#ef4444' : '#3b82f6';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }

    // 3. Draw Vector Road Network (SpaceNet 8 Graph)
    if (showRoads && (viewMode === 'vectors' || viewMode === 'hybrid') && result.road_network?.features) {
      result.road_network.features.forEach((feature) => {
        const coords = feature.geometry?.coordinates;
        if (!coords || coords.length < 2) return;

        const isFlooded = feature.properties?.flooded;
        ctx.beginPath();
        ctx.moveTo(coords[0][0], coords[0][1]);
        for (let i = 1; i < coords.length; i++) {
          ctx.lineTo(coords[i][0], coords[i][1]);
        }

        // Flooded = Crimson Red/Amber; Non-flooded = Emerald Green
        ctx.strokeStyle = isFlooded ? '#ef4444' : '#10b981';
        ctx.lineWidth = isFlooded ? 3.5 : 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
      });
    }

    // 4. Draw Calculated Flood-Safe Route
    if (routeResult?.route?.geometry?.coordinates?.length > 1) {
      const coords = routeResult.route.geometry.coordinates;
      const isFloodedRoute = routeResult.has_flooded_segments;

      // Glow effect
      ctx.shadowColor = isFloodedRoute ? '#ef4444' : '#06b6d4';
      ctx.shadowBlur = 12;

      ctx.beginPath();
      ctx.moveTo(coords[0][0], coords[0][1]);
      for (let i = 1; i < coords.length; i++) {
        ctx.lineTo(coords[i][0], coords[i][1]);
      }
      ctx.strokeStyle = isFloodedRoute ? '#f59e0b' : '#00f0ff';
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.stroke();

      ctx.shadowBlur = 0; // Reset shadow
    }

    // 5. Draw Start (A) and Destination (B) Markers
    if (startPoint) {
      ctx.beginPath();
      ctx.arc(startPoint.x, startPoint.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#06b6d4';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('A', startPoint.x, startPoint.y);
    }

    if (endPoint) {
      ctx.beginPath();
      ctx.arc(endPoint.x, endPoint.y, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 10px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('B', endPoint.x, endPoint.y);
    }

    ctx.restore();
  }, [result, showMask, showRoads, showBuildings, viewMode, overlayOpacity, routeResult, startPoint, endPoint, scale, offset, darkMode]);

  useEffect(() => {
    drawScene();
  }, [drawScene]);

  /* ─── Canvas Interaction Handlers ─── */
  const handleCanvasMouseDown = (e) => {
    if (e.button === 0 && !e.shiftKey && activeTab !== 'route') {
      setIsDragging(true);
      setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    }
  };

  const handleCanvasMouseMove = (e) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleCanvasMouseUp = () => {
    setIsDragging(false);
  };

  const handleCanvasWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => Math.min(Math.max(prev * zoomFactor, 0.3), 8));
  };

  const handleCanvasClick = (e) => {
    if (!result) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    const imgX = (clickX - (offset.x + centerX)) / scale + centerX;
    const imgY = (clickY - (offset.y + centerY)) / scale + centerY;

    if (activeTab === 'route') {
      if (routingMode === 'start' || !startPoint) {
        const pt = { x: imgX, y: imgY };
        setStartPoint(pt);
        setRoutingMode('end');
        if (endPoint) calculateRoute(pt, endPoint);
      } else {
        const pt = { x: imgX, y: imgY };
        setEndPoint(pt);
        setRoutingMode('start');
        calculateRoute(startPoint, pt);
      }
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
            <span className="brand-title">SpaceNet 8 Flood Mapping</span>
            <span className="brand-subtitle">Siamese Cross-Attention & Vector Routing</span>
          </div>
        </div>

        <div className="nav-tabs">
          {[
            { id: 'upload', label: '1. Upload & Inference', icon: '📡' },
            { id: 'results', label: '2. Damage Vector Paths', icon: '📊' },
            { id: 'route', label: '3. Safe Routing', icon: '🚗' },
          ].map((tab) => (
            <button
              key={tab.id}
              className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              <span>{tab.icon}</span> {tab.label}
            </button>
          ))}
        </div>

        <div className="nav-actions">
          <div
            className={`status-dot ${apiStatus?.status === 'ok' ? 'online' : 'offline'}`}
            title={apiStatus?.status === 'ok' ? 'Backend API Live' : 'Backend Disconnected'}
          />
          <button className="theme-btn" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? '☀ Light' : '🌙 Dark'}
          </button>
        </div>
      </nav>

      {/* ─ Main Layout ─ */}
      <div className="main-layout">
        {/* Left Sidebar */}
        <aside className="sidebar">
          {error && (
            <div className="alert alert-error">
              <span>⚠</span> {error}
              <button onClick={() => setError(null)}>×</button>
            </div>
          )}

          {/* TAB 1: UPLOAD */}
          {activeTab === 'upload' && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">📡</span>
                Satellite Image Pair
              </h2>
              <p className="panel-desc">
                Upload pre-disaster and post-flood GeoTIFFs to run multi-scale Siamese cross-attention change detection and extract vector damage paths.
              </p>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-badge pre">PRE</span> Pre-Event Image
                </label>
                <div className="upload-dropzone" onClick={() => document.getElementById('pre-input').click()}>
                  <input id="pre-input" type="file" accept=".tif,.tiff" hidden onChange={(e) => setPreFile(e.target.files[0])} />
                  {preFile ? <span className="upload-filename">✓ {preFile.name}</span> : <span className="upload-placeholder">Click to select Pre-event .tif</span>}
                </div>
              </div>

              <div className="upload-group">
                <label className="upload-label">
                  <span className="upload-badge post">POST</span> Post-Event Image
                </label>
                <div className="upload-dropzone" onClick={() => document.getElementById('post-input').click()}>
                  <input id="post-input" type="file" accept=".tif,.tiff" hidden onChange={(e) => setPostFile(e.target.files[0])} />
                  {postFile ? <span className="upload-filename">✓ {postFile.name}</span> : <span className="upload-placeholder">Click to select Post-event .tif</span>}
                </div>
              </div>

              <button className="btn btn-primary" onClick={handlePredict} disabled={loading}>
                {loading ? '⟳ Running Deep Inference...' : '🔍 Run Flood & Path Analysis'}
              </button>

              <div className="divider">
                <span>or</span>
              </div>

              <button className="btn btn-outline" onClick={handleSample} disabled={loading}>
                📋 Load SpaceNet 8 Sample Demo
              </button>
            </div>
          )}

          {/* TAB 2: RESULTS & DAMAGE METRICS */}
          {activeTab === 'results' && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">📊</span>
                Detection & Vector Summary
              </h2>

              {result ? (
                <>
                  <div className="vector-stats-badge">
                    <div><strong>{result.num_road_segments || 0}</strong> Road Segments</div>
                    <div><strong>{result.num_buildings || 0}</strong> Buildings</div>
                  </div>

                  <div className="results-grid">
                    {result.class_names?.map((name) => {
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

                  <button className="btn btn-primary" style={{ marginTop: '1.25rem' }} onClick={() => setActiveTab('route')}>
                    🚗 Test Safe Route Navigation →
                  </button>
                </>
              ) : (
                <div className="empty-notice">No prediction results yet. Please upload an image pair or load demo.</div>
              )}
            </div>
          )}

          {/* TAB 3: ROUTING */}
          {activeTab === 'route' && (
            <div className="panel">
              <h2 className="panel-title">
                <span className="panel-icon">🚗</span>
                Flood-Safe Dijkstra Router
              </h2>
              <p className="panel-desc">
                Click anywhere on the road paths to select <strong>Start (A)</strong> and <strong>Destination (B)</strong>. The routing engine navigates the graph avoiding flooded roads.
              </p>

              <div className="route-points">
                <div className={`route-point ${routingMode === 'start' ? 'active' : ''}`} onClick={() => setRoutingMode('start')}>
                  <span className="route-marker start">A</span>
                  <span>{startPoint ? `(${Math.round(startPoint.x)}, ${Math.round(startPoint.y)})` : 'Click road to set Start'}</span>
                </div>
                <div className={`route-point ${routingMode === 'end' ? 'active' : ''}`} onClick={() => setRoutingMode('end')}>
                  <span className="route-marker end">B</span>
                  <span>{endPoint ? `(${Math.round(endPoint.x)}, ${Math.round(endPoint.y)})` : 'Click road to set Destination'}</span>
                </div>
              </div>

              {routingLoading && <div className="loading-spinner">Calculating safe route along road network...</div>}

              {routeResult && (
                <div className="route-result">
                  <div className={`route-status ${routeResult.route_type}`}>
                    {routeResult.route_type === 'safe' ? '✅ Flood-Safe Route Found' : '⚠ Warning: Route Crosses Flood Zones'}
                  </div>
                  <div className="route-distance">
                    Total Distance: <strong>{routeResult.distance_m} meters</strong>
                  </div>
                  <div className="route-nodes">
                    Waypoints traversed: {routeResult.path_nodes?.length || 0} nodes
                  </div>
                </div>
              )}

              <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem' }}>
                <button
                  className="btn btn-outline"
                  onClick={() => {
                    setStartPoint(null);
                    setEndPoint(null);
                    setRouteResult(null);
                    setRoutingMode('start');
                  }}
                >
                  🔄 Clear Route
                </button>
              </div>
            </div>
          )}
        </aside>

        {/* Right Map / Vector Canvas Area */}
        <div className="map-view-container" ref={containerRef}>
          {/* Top Floating Control Bar */}
          <div className="floating-controls">
            <div className="control-group">
              <span className="control-label">View:</span>
              <button className={`control-btn ${viewMode === 'hybrid' ? 'active' : ''}`} onClick={() => setViewMode('hybrid')}>
                Hybrid
              </button>
              <button className={`control-btn ${viewMode === 'vectors' ? 'active' : ''}`} onClick={() => setViewMode('vectors')}>
                Vector Paths
              </button>
              <button className={`control-btn ${viewMode === 'raster' ? 'active' : ''}`} onClick={() => setViewMode('raster')}>
                Damage Mask
              </button>
            </div>

            <div className="control-group">
              <label className="toggle-label">
                <input type="checkbox" checked={showRoads} onChange={(e) => setShowRoads(e.target.checked)} />
                🛣 Roads
              </label>
              <label className="toggle-label">
                <input type="checkbox" checked={showBuildings} onChange={(e) => setShowBuildings(e.target.checked)} />
                🏢 Buildings
              </label>
              <label className="toggle-label">
                <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
                🌊 Flood
              </label>
            </div>

            <div className="control-group">
              <span className="control-label">Zoom:</span>
              <button className="control-btn" onClick={() => setScale((s) => Math.min(s * 1.25, 8))}>+</button>
              <button className="control-btn" onClick={() => setScale((s) => Math.max(s * 0.8, 0.3))}>-</button>
              <button className="control-btn" onClick={resetView}>Reset</button>
            </div>
          </div>

          {/* Interactive Canvas */}
          <div className="canvas-wrapper">
            <canvas
              ref={canvasRef}
              width={result?.mask_shape?.[2] || 512}
              height={result?.mask_shape?.[1] || 512}
              className="flood-canvas"
              onMouseDown={handleCanvasMouseDown}
              onMouseMove={handleCanvasMouseMove}
              onMouseUp={handleCanvasMouseUp}
              onWheel={handleCanvasWheel}
              onClick={handleCanvasClick}
            />

            {!result && (
              <div className="canvas-empty-state">
                <div className="empty-icon">🛰</div>
                <h3>SpaceNet 8 Interactive Map</h3>
                <p>Upload a Pre/Post GeoTIFF pair on the left or click <strong>Load SpaceNet 8 Sample Demo</strong> to view vector paths and flood damage.</p>
                <button className="btn btn-primary" onClick={handleSample}>
                  📋 Load SpaceNet 8 Sample Demo
                </button>
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="map-legend">
            <div className="legend-title">SpaceNet 8 Vector Legend</div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: '#10b981' }} />
              <span>Passable Road (Navigable)</span>
            </div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: '#ef4444' }} />
              <span>Flooded Road (Blocked)</span>
            </div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: '#3b82f6' }} />
              <span>Building Footprint</span>
            </div>
            <div className="legend-item">
              <div className="legend-swatch" style={{ background: '#00f0ff' }} />
              <span>Safe Evacuation Route</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
