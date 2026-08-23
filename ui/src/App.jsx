import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const METRIC_COLORS = {
  building: { text: '#2563eb', bg: '#eff6ff', border: '#bfdbfe' },
  road: { text: '#475569', bg: '#f8fafc', border: '#e2e8f0' },
  flood: { text: '#0284c7', bg: '#f0f9ff', border: '#bae6fd' },
  flooded_building: { text: '#dc2626', bg: '#fef2f2', border: '#fecaca' },
  flooded_road: { text: '#d97706', bg: '#fffbeb', border: '#fde68a' },
};

export default function App() {
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // 'overview' | 'side-by-side' | 'router'
  const [apiStatus, setApiStatus] = useState(null);

  // Visualization toggles
  const [showRoads, setShowRoads] = useState(true);
  const [showBuildings, setShowBuildings] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [showSatellite, setShowSatellite] = useState(true);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);

  // Routing State
  const [startPoint, setStartPoint] = useState(null);
  const [endPoint, setEndPoint] = useState(null);
  const [routeResult, setRouteResult] = useState(null);
  const [routingLoading, setRoutingLoading] = useState(false);
  const [routeStep, setRouteStep] = useState('start'); // 'start' | 'end'

  // Canvas Pan & Zoom
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const mainCanvasRef = useRef(null);
  const preCanvasRef = useRef(null);
  const postCanvasRef = useRef(null);

  // Images cache
  const preImgObj = useRef(null);
  const postImgObj = useRef(null);
  const maskImgObj = useRef(null);

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

  /* ─── Load Model Predictions ─── */
  const handlePredict = async () => {
    if (!preFile || !postFile) {
      setError('Please select both Pre-event and Post-event GeoTIFF images.');
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
        resetView();
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference failed. Verify backend status.');
    } finally {
      setLoading(false);
    }
  };

  /* ─── Route Calculation (Dijkstra) ─── */
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
      setError('Route calculation error.');
    } finally {
      setRoutingLoading(false);
    }
  };

  const resetView = () => {
    setScale(1);
    setOffset({ x: 0, y: 0 });
  };

  /* ─── Pre-load Images into Memory ─── */
  useEffect(() => {
    if (result?.pre_image_base64) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${result.pre_image_base64}`;
      preImgObj.current = img;
    } else {
      preImgObj.current = null;
    }

    if (result?.post_image_base64) {
      const img = new Image();
      img.src = `data:image/jpeg;base64,${result.post_image_base64}`;
      postImgObj.current = img;
    } else {
      postImgObj.current = null;
    }

    if (result?.mask_png_base64) {
      const img = new Image();
      img.src = `data:image/png;base64,${result.mask_png_base64}`;
      maskImgObj.current = img;
    } else {
      maskImgObj.current = null;
    }
  }, [result]);

  /* ─── Main Unified Canvas Renderer ─── */
  const drawMainCanvas = useCallback(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas || !result) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);
    ctx.save();
    ctx.translate(offset.x + width / 2, offset.y + height / 2);
    ctx.scale(scale, scale);
    ctx.translate(-width / 2, -height / 2);

    const maskShape = result.mask_shape || [5, 512, 512];
    const H = maskShape[1] || 512;
    const W = maskShape[2] || 512;

    // Base background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);

    // 1. Satellite Base Image
    if (showSatellite && postImgObj.current?.complete) {
      ctx.drawImage(postImgObj.current, 0, 0, W, H);
    } else if (showSatellite && preImgObj.current?.complete) {
      ctx.drawImage(preImgObj.current, 0, 0, W, H);
    }

    // 2. Semantic Damage Mask
    if (showMask && maskImgObj.current?.complete) {
      ctx.globalAlpha = overlayOpacity;
      ctx.drawImage(maskImgObj.current, 0, 0, W, H);
      ctx.globalAlpha = 1.0;
    }

    // 3. Vector Building Footprints (Pre vs Post)
    if (showBuildings && result.building_footprints?.features) {
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

        ctx.fillStyle = isFlooded ? 'rgba(239, 68, 68, 0.45)' : 'rgba(37, 99, 235, 0.35)';
        ctx.fill();
        ctx.strokeStyle = isFlooded ? '#ef4444' : '#2563eb';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      });
    }

    // 4. SpaceNet 8 Vector Road Graph Lines
    if (showRoads && result.road_network?.features) {
      result.road_network.features.forEach((feature) => {
        const coords = feature.geometry?.coordinates;
        if (!coords || coords.length < 2) return;

        const isFlooded = feature.properties?.flooded;
        ctx.beginPath();
        ctx.moveTo(coords[0][0], coords[0][1]);
        for (let i = 1; i < coords.length; i++) {
          ctx.lineTo(coords[i][0], coords[i][1]);
        }

        // Green for flood-free passable road, Red for flooded/blocked road
        ctx.strokeStyle = isFlooded ? '#dc2626' : '#059669';
        ctx.lineWidth = isFlooded ? 3.5 : 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();
      });
    }

    // 5. Dijkstra Safe Navigation Path
    if (routeResult?.route?.geometry?.coordinates?.length > 1) {
      const coords = routeResult.route.geometry.coordinates;
      const isFloodedRoute = routeResult.has_flooded_segments;

      ctx.shadowColor = isFloodedRoute ? '#dc2626' : '#06b6d4';
      ctx.shadowBlur = 10;

      ctx.beginPath();
      ctx.moveTo(coords[0][0], coords[0][1]);
      for (let i = 1; i < coords.length; i++) {
        ctx.lineTo(coords[i][0], coords[i][1]);
      }
      ctx.strokeStyle = isFloodedRoute ? '#f59e0b' : '#00f0ff';
      ctx.lineWidth = 4.5;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    // 6. Start (A) & Destination (B) Markers
    if (startPoint) {
      ctx.beginPath();
      ctx.arc(startPoint.x, startPoint.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#0284c7';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('A', startPoint.x, startPoint.y);
    }

    if (endPoint) {
      ctx.beginPath();
      ctx.arc(endPoint.x, endPoint.y, 7, 0, Math.PI * 2);
      ctx.fillStyle = '#d97706';
      ctx.fill();
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = 'bold 9px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('B', endPoint.x, endPoint.y);
    }

    ctx.restore();
  }, [result, showSatellite, showMask, showBuildings, showRoads, overlayOpacity, routeResult, startPoint, endPoint, scale, offset]);

  /* ─── Side-by-Side Pre / Post Canvas Renderers ─── */
  const drawSideBySide = useCallback(() => {
    if (!result) return;
    const maskShape = result.mask_shape || [5, 512, 512];
    const H = maskShape[1] || 512;
    const W = maskShape[2] || 512;

    // 1. Pre Canvas (Before Disaster)
    const preCanvas = preCanvasRef.current;
    if (preCanvas) {
      const ctx = preCanvas.getContext('2d');
      ctx.clearRect(0, 0, preCanvas.width, preCanvas.height);
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, W, H);

      if (preImgObj.current?.complete) {
        ctx.drawImage(preImgObj.current, 0, 0, W, H);
      }

      // Draw intact buildings & roads
      if (result.building_footprints?.features) {
        result.building_footprints.features.forEach((feature) => {
          const coords = feature.geometry?.coordinates?.[0];
          if (!coords || coords.length < 3) return;
          ctx.beginPath();
          ctx.moveTo(coords[0][0], coords[0][1]);
          for (let i = 1; i < coords.length; i++) ctx.lineTo(coords[i][0], coords[i][1]);
          ctx.closePath();
          ctx.fillStyle = 'rgba(37, 99, 235, 0.35)';
          ctx.fill();
          ctx.strokeStyle = '#2563eb';
          ctx.lineWidth = 1.2;
          ctx.stroke();
        });
      }

      if (result.road_network?.features) {
        result.road_network.features.forEach((feature) => {
          const coords = feature.geometry?.coordinates;
          if (!coords || coords.length < 2) return;
          ctx.beginPath();
          ctx.moveTo(coords[0][0], coords[0][1]);
          for (let i = 1; i < coords.length; i++) ctx.lineTo(coords[i][0], coords[i][1]);
          ctx.strokeStyle = '#059669';
          ctx.lineWidth = 2.0;
          ctx.stroke();
        });
      }
    }

    // 2. Post Canvas (After Flood Disaster)
    const postCanvas = postCanvasRef.current;
    if (postCanvas) {
      const ctx = postCanvas.getContext('2d');
      ctx.clearRect(0, 0, postCanvas.width, postCanvas.height);
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, W, H);

      if (postImgObj.current?.complete) {
        ctx.drawImage(postImgObj.current, 0, 0, W, H);
      }

      // Damage mask overlay
      if (maskImgObj.current?.complete) {
        ctx.globalAlpha = 0.65;
        ctx.drawImage(maskImgObj.current, 0, 0, W, H);
        ctx.globalAlpha = 1.0;
      }

      // Flooded buildings
      if (result.building_footprints?.features) {
        result.building_footprints.features.forEach((feature) => {
          const coords = feature.geometry?.coordinates?.[0];
          if (!coords || coords.length < 3) return;
          const isFlooded = feature.properties?.flooded;
          ctx.beginPath();
          ctx.moveTo(coords[0][0], coords[0][1]);
          for (let i = 1; i < coords.length; i++) ctx.lineTo(coords[i][0], coords[i][1]);
          ctx.closePath();
          ctx.fillStyle = isFlooded ? 'rgba(220, 38, 38, 0.5)' : 'rgba(37, 99, 235, 0.2)';
          ctx.fill();
          ctx.strokeStyle = isFlooded ? '#dc2626' : '#2563eb';
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      }

      // Flooded roads
      if (result.road_network?.features) {
        result.road_network.features.forEach((feature) => {
          const coords = feature.geometry?.coordinates;
          if (!coords || coords.length < 2) return;
          const isFlooded = feature.properties?.flooded;
          ctx.beginPath();
          ctx.moveTo(coords[0][0], coords[0][1]);
          for (let i = 1; i < coords.length; i++) ctx.lineTo(coords[i][0], coords[i][1]);
          ctx.strokeStyle = isFlooded ? '#dc2626' : '#059669';
          ctx.lineWidth = isFlooded ? 3.0 : 1.8;
          ctx.stroke();
        });
      }
    }
  }, [result]);

  useEffect(() => {
    if (activeTab === 'side-by-side') {
      drawSideBySide();
    } else {
      drawMainCanvas();
    }
  }, [activeTab, drawMainCanvas, drawSideBySide]);

  /* ─── Canvas Mouse / Drag Handlers ─── */
  const handleMouseDown = (e) => {
    if (e.button === 0 && activeTab !== 'router') {
      setIsDragging(true);
      setDragStart({ x: e.clientX - offset.x, y: e.clientY - offset.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setOffset({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleWheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setScale((prev) => Math.min(Math.max(prev * zoomFactor, 0.4), 6.0));
  };

  const handleCanvasClick = (e) => {
    if (!result || activeTab !== 'router') return;
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();

    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    const imgX = (clickX - (offset.x + centerX)) / scale + centerX;
    const imgY = (clickY - (offset.y + centerY)) / scale + centerY;

    if (routeStep === 'start' || !startPoint) {
      const pt = { x: imgX, y: imgY };
      setStartPoint(pt);
      setRouteStep('end');
      if (endPoint) calculateRoute(pt, endPoint);
    } else {
      const pt = { x: imgX, y: imgY };
      setEndPoint(pt);
      setRouteStep('start');
      calculateRoute(startPoint, pt);
    }
  };

  return (
    <div className="swiss-app">
      {/* ── Header ── */}
      <header className="swiss-header">
        <div className="header-left">
          <div className="swiss-logo">
            <span className="logo-badge">SN8</span>
          </div>
          <div>
            <h1 className="swiss-title">SpaceNet 8 Flood Intelligence</h1>
            <p className="swiss-subtitle">Siamese Cross-Attention & Vector Routing Platform</p>
          </div>
        </div>

        {/* Segmented Navigation Tabs */}
        <div className="segmented-tabs">
          <button
            className={`tab-pill ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            🗺 Interactive Map
          </button>
          <button
            className={`tab-pill ${activeTab === 'side-by-side' ? 'active' : ''}`}
            onClick={() => setActiveTab('side-by-side')}
          >
            ⚖ Before / After
          </button>
          <button
            className={`tab-pill ${activeTab === 'router' ? 'active' : ''}`}
            onClick={() => setActiveTab('router')}
          >
            🚗 Safe Route Planner
          </button>
        </div>

        <div className="header-right">
          <div className="api-badge" title="API Status">
            <span className={`status-indicator ${apiStatus?.status === 'ok' ? 'online' : 'offline'}`} />
            <span>{apiStatus?.status === 'ok' ? 'API Online' : 'API Offline'}</span>
          </div>
        </div>
      </header>

      {/* ── Main Layout ── */}
      <div className="swiss-body">
        {/* Left Control & Metrics Sidebar */}
        <aside className="swiss-sidebar">
          {error && (
            <div className="swiss-alert error">
              <span>⚠</span> {error}
              <button className="alert-close" onClick={() => setError(null)}>×</button>
            </div>
          )}

          {/* Upload Section */}
          <div className="card">
            <div className="card-header">
              <span className="card-tag">INPUTS</span>
              <h2 className="card-title">Satellite Imagery</h2>
            </div>

            <div className="upload-row">
              <div className="file-box" onClick={() => document.getElementById('pre-upload').click()}>
                <input id="pre-upload" type="file" accept=".tif,.tiff" hidden onChange={(e) => setPreFile(e.target.files[0])} />
                <span className="file-tag pre">PRE</span>
                <span className="file-name">{preFile ? preFile.name : 'Select Pre-event .tif'}</span>
              </div>

              <div className="file-box" onClick={() => document.getElementById('post-upload').click()}>
                <input id="post-upload" type="file" accept=".tif,.tiff" hidden onChange={(e) => setPostFile(e.target.files[0])} />
                <span className="file-tag post">POST</span>
                <span className="file-name">{postFile ? postFile.name : 'Select Post-event .tif'}</span>
              </div>
            </div>

            <button className="btn-primary" onClick={handlePredict} disabled={loading || !preFile || !postFile}>
              {loading ? '⟳ Running Siamese Inference...' : 'Analyze Flood Damage'}
            </button>
          </div>

          {/* Results Summary */}
          {result && (
            <div className="card">
              <div className="card-header">
                <span className="card-tag">DETECTION</span>
                <h2 className="card-title">Damage Telemetry</h2>
              </div>

              <div className="stat-pill-row">
                <div className="stat-pill">
                  <span className="stat-num">{result.num_road_segments || 0}</span>
                  <span className="stat-lbl">Road Vectors</span>
                </div>
                <div className="stat-pill">
                  <span className="stat-num">{result.num_buildings || 0}</span>
                  <span className="stat-lbl">Buildings</span>
                </div>
              </div>

              <div className="metrics-list">
                {result.class_names?.map((name) => {
                  const stat = result.statistics?.[name];
                  if (!stat) return null;
                  const c = METRIC_COLORS[name] || METRIC_COLORS.road;
                  return (
                    <div key={name} className="metric-row">
                      <div className="metric-info">
                        <span className="metric-dot" style={{ background: c.text }} />
                        <span className="metric-name">{name.replace('_', ' ')}</span>
                      </div>
                      <div className="metric-val" style={{ color: c.text }}>
                        {stat.coverage_pct}%
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Routing Controller */}
          {activeTab === 'router' && (
            <div className="card">
              <div className="card-header">
                <span className="card-tag">NAVIGATION</span>
                <h2 className="card-title">Dijkstra Safe Router</h2>
              </div>
              <p className="card-desc">Click two road intersections on the map to compute flood-safe shortest path.</p>

              <div className="route-pickers">
                <div className={`picker-box ${routeStep === 'start' ? 'active' : ''}`} onClick={() => setRouteStep('start')}>
                  <span className="picker-badge a">A</span>
                  <span>{startPoint ? `(${Math.round(startPoint.x)}, ${Math.round(startPoint.y)})` : 'Click map to set Start'}</span>
                </div>
                <div className={`picker-box ${routeStep === 'end' ? 'active' : ''}`} onClick={() => setRouteStep('end')}>
                  <span className="picker-badge b">B</span>
                  <span>{endPoint ? `(${Math.round(endPoint.x)}, ${Math.round(endPoint.y)})` : 'Click map to set End'}</span>
                </div>
              </div>

              {routingLoading && <div className="loading-bar">Computing shortest safe path along network...</div>}

              {routeResult && (
                <div className="route-report">
                  <div className={`route-status ${routeResult.route_type}`}>
                    {routeResult.route_type === 'safe' ? '✓ Safe Flood-Free Route' : '⚠ Route Uses Flooded Segment'}
                  </div>
                  <div className="route-stat">
                    <span>Distance:</span> <strong>{routeResult.distance_m} meters</strong>
                  </div>
                  <div className="route-stat">
                    <span>Waypoints:</span> <strong>{routeResult.path_nodes?.length || 0} nodes</strong>
                  </div>
                </div>
              )}

              <button
                className="btn-secondary"
                style={{ marginTop: '0.75rem' }}
                onClick={() => {
                  setStartPoint(null);
                  setEndPoint(null);
                  setRouteResult(null);
                  setRouteStep('start');
                }}
              >
                Clear Route
              </button>
            </div>
          )}
        </aside>

        {/* Main Visual Viewport */}
        <main className="swiss-viewport">
          {/* Top Layer & View Controls */}
          <div className="viewport-controls">
            <div className="toggle-group">
              <label className="toggle-chip">
                <input type="checkbox" checked={showSatellite} onChange={(e) => setShowSatellite(e.target.checked)} />
                Satellite Image
              </label>
              <label className="toggle-chip">
                <input type="checkbox" checked={showRoads} onChange={(e) => setShowRoads(e.target.checked)} />
                Road Vectors
              </label>
              <label className="toggle-chip">
                <input type="checkbox" checked={showBuildings} onChange={(e) => setShowBuildings(e.target.checked)} />
                Buildings
              </label>
              <label className="toggle-chip">
                <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
                Damage Mask
              </label>
            </div>

            <div className="zoom-controls">
              <span className="zoom-lbl">Zoom</span>
              <button className="btn-zoom" onClick={() => setScale((s) => Math.min(s * 1.2, 5))}>+</button>
              <button className="btn-zoom" onClick={() => setScale((s) => Math.max(s * 0.8, 0.4))}>-</button>
              <button className="btn-zoom" onClick={resetView}>Reset</button>
            </div>
          </div>

          {/* VIEW MODE 1: Interactive Unified Map or Router */}
          {activeTab !== 'side-by-side' && (
            <div className="canvas-stage">
              <canvas
                ref={mainCanvasRef}
                width={result?.mask_shape?.[2] || 512}
                height={result?.mask_shape?.[1] || 512}
                className="main-canvas"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onWheel={handleWheel}
                onClick={handleCanvasClick}
              />

              {!result && (
                <div className="empty-placeholder">
                  <div className="empty-circle">🛰</div>
                  <h3>SpaceNet 8 Geospatial Intelligence</h3>
                  <p>Upload a Pre-event and Post-event GeoTIFF image pair on the left to run deep flood change detection and vector routing.</p>
                </div>
              )}
            </div>
          )}

          {/* VIEW MODE 2: Side-by-Side Before vs After Comparison */}
          {activeTab === 'side-by-side' && (
            <div className="side-by-side-grid">
              <div className="comparison-pane">
                <div className="pane-header">
                  <span className="pane-badge pre">PRE-EVENT</span>
                  <span>Intact Infrastructure (Before)</span>
                </div>
                <div className="pane-canvas-box">
                  <canvas
                    ref={preCanvasRef}
                    width={result?.mask_shape?.[2] || 512}
                    height={result?.mask_shape?.[1] || 512}
                    className="side-canvas"
                  />
                </div>
              </div>

              <div className="comparison-pane">
                <div className="pane-header">
                  <span className="pane-badge post">POST-EVENT</span>
                  <span>Flood Inundation & Damage (After)</span>
                </div>
                <div className="pane-canvas-box">
                  <canvas
                    ref={postCanvasRef}
                    width={result?.mask_shape?.[2] || 512}
                    height={result?.mask_shape?.[1] || 512}
                    className="side-canvas"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Modern Minimal Legend */}
          <div className="swiss-legend">
            <div className="legend-row">
              <span className="swatch" style={{ background: '#059669' }} />
              <span>Passable Road</span>
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#dc2626' }} />
              <span>Flooded Road</span>
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#2563eb' }} />
              <span>Building</span>
            </div>
            <div className="legend-row">
              <span className="swatch" style={{ background: '#00f0ff' }} />
              <span>Safe Evacuation Route</span>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
