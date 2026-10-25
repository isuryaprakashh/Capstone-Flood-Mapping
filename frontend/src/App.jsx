import React, { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE = '/api';

const COLORS = {
  building: '#2563eb',
  floodedBuilding: '#dc2626',
  road: '#059669',
  floodedRoad: '#d97706',
  groundTruth: '#facc15',
  groundTruthFlooded: '#f97316',
};

const ACCURACY_ROWS = [
  ['building', 'Building'],
  ['road', 'Road'],
  ['flooded_building', 'Flooded building'],
  ['flooded_road', 'Flooded road'],
];

const INITIAL_VIEW = { scale: 1, x: 0, y: 0 };
const PAGE_SIZE = 20;

/* ─── Load a base64 image into an <img> element; re-renders when ready ─── */
function useDataImage(b64, mime) {
  const [img, setImg] = useState(null);
  useEffect(() => {
    if (!b64) {
      setImg(null);
      return undefined;
    }
    let alive = true;
    const el = new Image();
    el.onload = () => alive && setImg(el);
    el.src = `data:${mime};base64,${b64}`;
    return () => {
      alive = false;
    };
  }, [b64, mime]);
  return img;
}

/* ─── Canvas drawing ─── */
function tracePolygon(ctx, geometry) {
  const polygons = geometry?.type === 'MultiPolygon' ? geometry.coordinates : [geometry?.coordinates];
  polygons.forEach((rings) => {
    const ring = rings?.[0];
    if (!ring || ring.length < 3) return;
    ctx.moveTo(ring[0][0], ring[0][1]);
    for (let i = 1; i < ring.length; i++) ctx.lineTo(ring[i][0], ring[i][1]);
    ctx.closePath();
  });
}

function traceLine(ctx, geometry) {
  const lines = geometry?.type === 'MultiLineString' ? geometry.coordinates : [geometry?.coordinates];
  lines.forEach((coords) => {
    if (!coords || coords.length < 2) return;
    ctx.moveTo(coords[0][0], coords[0][1]);
    for (let i = 1; i < coords.length; i++) ctx.lineTo(coords[i][0], coords[i][1]);
  });
}

function drawScene(canvas, { base, mask, result, layers, view }) {
  if (!canvas || !result) return;
  const ctx = canvas.getContext('2d');
  const [, H, W] = result.mask_shape;
  // Keep strokes a constant on-screen width regardless of zoom and canvas scaling
  const px = (W / (canvas.clientWidth || W)) / view.scale;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.setTransform(view.scale, 0, 0, view.scale, view.x, view.y);

  if (base) ctx.drawImage(base, 0, 0, W, H);

  if (layers.mask && mask) {
    ctx.globalAlpha = layers.opacity;
    ctx.drawImage(mask, 0, 0, W, H);
    ctx.globalAlpha = 1;
  }

  if (layers.buildings) {
    result.building_footprints?.features?.forEach((f) => {
      const flooded = f.properties?.flooded;
      ctx.beginPath();
      tracePolygon(ctx, f.geometry);
      ctx.fillStyle = flooded ? 'rgba(220, 38, 38, 0.40)' : 'rgba(37, 99, 235, 0.28)';
      ctx.fill();
      ctx.strokeStyle = flooded ? COLORS.floodedBuilding : COLORS.building;
      ctx.lineWidth = 1.4 * px;
      ctx.stroke();
    });
  }

  if (layers.roads) {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    result.road_network?.features?.forEach((f) => {
      const flooded = f.properties?.flooded;
      ctx.beginPath();
      traceLine(ctx, f.geometry);
      ctx.strokeStyle = flooded ? COLORS.floodedRoad : COLORS.road;
      ctx.lineWidth = (flooded ? 3.2 : 2.4) * px;
      ctx.stroke();
    });
  }

  const gt = result.ground_truth;
  if (layers.groundTruth && gt) {
    ctx.setLineDash([4 * px, 3 * px]);
    gt.building_footprints?.features?.forEach((f) => {
      ctx.beginPath();
      tracePolygon(ctx, f.geometry);
      ctx.strokeStyle = f.properties?.flooded ? COLORS.groundTruthFlooded : COLORS.groundTruth;
      ctx.lineWidth = 1.4 * px;
      ctx.stroke();
    });
    gt.road_network?.features?.forEach((f) => {
      ctx.beginPath();
      traceLine(ctx, f.geometry);
      ctx.strokeStyle = f.properties?.flooded ? COLORS.groundTruthFlooded : COLORS.groundTruth;
      ctx.lineWidth = 1.6 * px;
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }
}

/* ─── Small presentational pieces ─── */
function StatTile({ label, value, sub, tone }) {
  return (
    <div className="stat-tile">
      <span className="stat-label">{label}</span>
      <span className={`stat-value ${tone || ''}`}>{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

function LayerChip({ active, onClick, swatch, title, children }) {
  return (
    <button className={`chip ${active ? 'active' : ''}`} onClick={onClick} aria-pressed={active} title={title}>
      {swatch && <i className="chip-swatch" style={{ background: swatch }} />}
      {children}
    </button>
  );
}

const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}`);

// "Louisiana-East_Training_Public" → "Louisiana-East"
const aoiName = (aoi) => aoi.split('_')[0];

export default function App() {
  const [apiStatus, setApiStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Input: pick a mapped pair from the dataset, or upload files
  const [inputMode, setInputMode] = useState('dataset'); // 'dataset' | 'upload'
  const [aois, setAois] = useState([]);
  const [aoi, setAoi] = useState('');
  const [query, setQuery] = useState('');
  const [page, setPage] = useState(0);
  const [pairResults, setPairResults] = useState({ pairs: [], total: 0 });
  const pairListRef = useRef(null);
  const [selectedPair, setSelectedPair] = useState(null);
  const [selectedPost, setSelectedPost] = useState('');
  const [preFile, setPreFile] = useState(null);
  const [postFile, setPostFile] = useState(null);
  const [autoMatch, setAutoMatch] = useState(null); // upload mode: { status, post, aoi, shift }

  // View (before/after panes share layers, zoom and pan)
  const [layers, setLayers] = useState({
    buildings: true,
    roads: true,
    mask: false,
    groundTruth: false,
    opacity: 0.6,
  });
  const [view, setView] = useState(INITIAL_VIEW);
  const dragRef = useRef(null);

  const preCanvasRef = useRef(null);
  const postCanvasRef = useRef(null);

  const preImg = useDataImage(result?.pre_image_base64, 'image/jpeg');
  const postImg = useDataImage(result?.post_image_base64, 'image/jpeg');
  const maskImg = useDataImage(result?.mask_png_base64, 'image/png');

  /* ─── API status & dataset index ─── */
  useEffect(() => {
    axios
      .get(`${API_BASE}/health`)
      .then((res) => setApiStatus(res.data))
      .catch(() => setApiStatus({ status: 'error' }));

    axios
      .get(`${API_BASE}/pairs/aois`)
      .then((res) => {
        const list = res.data.aois || [];
        setAois(list);
        if (list.length) setAoi((list.find((a) => a.has_labels) || list[0]).aoi);
      })
      .catch(() => setAois([]));
  }, []);

  useEffect(() => {
    if (!aoi) return undefined;
    const timer = setTimeout(() => {
      axios
        .get(`${API_BASE}/pairs`, {
          params: { aoi, q: query || undefined, limit: PAGE_SIZE, offset: page * PAGE_SIZE },
        })
        .then((res) => {
          setPairResults({ pairs: res.data.pairs || [], total: res.data.total || 0 });
          if (pairListRef.current) pairListRef.current.scrollTop = 0;
        })
        .catch(() => setPairResults({ pairs: [], total: 0 }));
    }, 250);
    return () => clearTimeout(timer);
  }, [aoi, query, page]);

  const pageCount = Math.max(1, Math.ceil(pairResults.total / PAGE_SIZE));

  const selectPair = (pair) => {
    setSelectedPair(pair);
    setSelectedPost(pair.post_images[0] || '');
  };

  /* ─── Upload mode: auto-attach the mapped post image for a chosen pre image ─── */
  const handlePreFileChange = async (file) => {
    setPreFile(file || null);
    setAutoMatch(null);
    if (!file) return;

    setAutoMatch({ status: 'loading' });
    try {
      const { data } = await axios.get(`${API_BASE}/pairs/lookup`, { params: { pre_image: file.name } });
      const pair = data.pair;
      const postName = pair.post_images[0];
      const res = await axios.get(`${API_BASE}/pairs/image`, {
        params: { aoi: pair.aoi, kind: 'post', name: postName },
        responseType: 'blob',
      });
      setPostFile(new File([res.data], postName, { type: 'image/tiff' }));
      setAutoMatch({ status: 'matched', post: postName, aoi: pair.aoi, shift: pair.post_shift_px?.[0] });
    } catch (err) {
      setAutoMatch({ status: err.response?.status === 404 ? 'none' : 'error' });
    }
  };

  const handlePostFileChange = (file) => {
    setPostFile(file || null);
    if (file) setAutoMatch({ status: 'manual', post: file.name });
  };

  /* ─── Prediction ─── */
  const handlePredict = async () => {
    const form = new FormData();
    let url = `${API_BASE}/predict`;

    if (inputMode === 'dataset') {
      if (!selectedPair) return;
      url = `${API_BASE}/predict_pair`;
      form.append('aoi', selectedPair.aoi);
      form.append('pre_image', selectedPair.pre_image);
      if (selectedPost) form.append('post_image', selectedPost);
    } else {
      if (!preFile || !postFile) return;
      form.append('pre_image', preFile);
      form.append('post_image', postFile);
    }

    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(url, form, { headers: { 'Content-Type': 'multipart/form-data' } });
      if (res.data.ok) {
        setResult(res.data);
        setView(INITIAL_VIEW);
        setLayers((l) => ({ ...l, groundTruth: false }));
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Inference failed. Verify backend status.');
    } finally {
      setLoading(false);
    }
  };

  const canPredict = inputMode === 'dataset' ? !!selectedPair : !!(preFile && postFile);

  /* ─── Rendering ─── */
  useEffect(() => {
    if (!result) return;
    const shared = { mask: maskImg, result, layers, view };
    drawScene(preCanvasRef.current, { ...shared, base: preImg });
    drawScene(postCanvasRef.current, { ...shared, base: postImg });
  }, [result, preImg, postImg, maskImg, layers, view]);

  /* ─── Pan & zoom (shared by both panes) ─── */
  const toCanvas = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const k = e.currentTarget.width / rect.width;
    return { x: (e.clientX - rect.left) * k, y: (e.clientY - rect.top) * k, k };
  };

  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    const p = toCanvas(e);
    dragRef.current = { x: p.x, y: p.y, view };
  };

  const handleMouseMove = (e) => {
    if (!dragRef.current) return;
    const p = toCanvas(e);
    const start = dragRef.current;
    setView({ ...start.view, x: start.view.x + (p.x - start.x), y: start.view.y + (p.y - start.y) });
  };

  const endDrag = () => {
    dragRef.current = null;
  };

  const zoomAt = useCallback((cx, cy, factor) => {
    setView((v) => {
      const scale = Math.min(Math.max(v.scale * factor, 1), 12);
      const r = scale / v.scale;
      return { scale, x: cx - (cx - v.x) * r, y: cy - (cy - v.y) * r };
    });
  }, []);

  const handleWheel = (e) => {
    const p = toCanvas(e);
    zoomAt(p.x, p.y, e.deltaY < 0 ? 1.15 : 1 / 1.15);
  };

  // React's onWheel is passive; block page scroll while zooming the canvas
  useEffect(() => {
    const stop = (e) => e.target.tagName === 'CANVAS' && e.preventDefault();
    window.addEventListener('wheel', stop, { passive: false });
    return () => window.removeEventListener('wheel', stop);
  }, []);

  const zoomCenter = (factor) => {
    const W = result?.mask_shape?.[2] || 512;
    const H = result?.mask_shape?.[1] || 512;
    zoomAt(W / 2, H / 2, factor);
  };

  const canvasProps = {
    width: result?.mask_shape?.[2] || 512,
    height: result?.mask_shape?.[1] || 512,
    onMouseDown: handleMouseDown,
    onMouseMove: handleMouseMove,
    onMouseUp: endDrag,
    onMouseLeave: endDrag,
    onWheel: handleWheel,
  };

  const summary = result?.summary;
  const gt = result?.ground_truth;
  const toggle = (key) => setLayers((l) => ({ ...l, [key]: !l[key] }));

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app-header">
        <div className="brand">
          <span className="brand-badge">SN8</span>
          <div>
            <h1 className="brand-title">SpaceNet 8 Flood Mapping</h1>
            <p className="brand-subtitle">Flooded building &amp; road detection from pre/post-event satellite imagery</p>
          </div>
        </div>

        <div className="header-status">
          {apiStatus?.model && (
            <span className={`model-badge ${apiStatus.model.stub ? 'stub' : ''}`} title={apiStatus.model.checkpoint || ''}>
              {apiStatus.model.name}
            </span>
          )}
          <span className="api-badge">
            <span className={`status-dot ${apiStatus?.status === 'ok' ? 'online' : 'offline'}`} />
            {apiStatus?.status === 'ok' ? 'API online' : 'API offline'}
          </span>
        </div>
      </header>

      <div className="app-body">
        {/* ── Left: input ── */}
        <aside className="panel panel-input">
          {error && (
            <div className="alert">
              <span>{error}</span>
              <button className="alert-close" onClick={() => setError(null)} aria-label="Dismiss">×</button>
            </div>
          )}

          <section className="card">
            <div className="card-head">
              <span className="card-tag">INPUT</span>
              <h2 className="card-title">Image pair</h2>
            </div>

            <div className="segmented">
              <button className={inputMode === 'dataset' ? 'active' : ''} onClick={() => setInputMode('dataset')}>
                From dataset
              </button>
              <button className={inputMode === 'upload' ? 'active' : ''} onClick={() => setInputMode('upload')}>
                Upload files
              </button>
            </div>

            {inputMode === 'dataset' ? (
              <div className="stack">
                <select
                  className="field"
                  value={aoi}
                  onChange={(e) => {
                    setAoi(e.target.value);
                    setPage(0);
                    setSelectedPair(null);
                  }}
                >
                  {aois.map((a) => (
                    <option key={a.aoi} value={a.aoi}>
                      {aoiName(a.aoi)} · {a.num_pairs} tiles · {a.has_labels ? 'labelled' : 'unlabelled test set'}
                    </option>
                  ))}
                </select>

                <input
                  className="field"
                  type="search"
                  placeholder="Search pre-event tile (e.g. 0_15_63)"
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    setPage(0);
                  }}
                />

                <div className="pair-list" role="listbox" ref={pairListRef}>
                  {pairResults.pairs.map((p) => (
                    <button
                      key={p.pre_image}
                      className={`pair-item ${selectedPair?.pre_image === p.pre_image ? 'active' : ''}`}
                      onClick={() => selectPair(p)}
                      title={p.pre_image}
                    >
                      <span className="mono">{p.label.replace('.geojson', '') || p.pre_image}</span>
                      <span className="muted">{p.post_images.length > 1 ? '2 post images' : '1 post image'}</span>
                    </button>
                  ))}
                  {pairResults.pairs.length === 0 && <div className="pair-empty">No tiles match.</div>}
                </div>
                {pairResults.total > 0 && (
                  <div className="pager">
                    <button onClick={() => setPage((p) => Math.max(0, p - 1))} disabled={page === 0} aria-label="Previous page">
                      ‹ Prev
                    </button>
                    <span className="pager-info">
                      {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, pairResults.total)} of {pairResults.total}
                      <span className="muted"> · page {page + 1}/{pageCount}</span>
                    </span>
                    <button
                      onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
                      disabled={page >= pageCount - 1}
                      aria-label="Next page"
                    >
                      Next ›
                    </button>
                  </div>
                )}

                {selectedPair && (
                  <div className="stack">
                    <div className="thumb-row">
                      <figure className="thumb">
                        <img
                          src={`${API_BASE}/pairs/thumbnail?aoi=${encodeURIComponent(selectedPair.aoi)}&kind=pre&name=${encodeURIComponent(selectedPair.pre_image)}`}
                          alt="Pre-event preview"
                        />
                        <figcaption><span className="tag pre">PRE</span>{selectedPair.pre_image}</figcaption>
                      </figure>
                      <figure className="thumb">
                        <img
                          src={`${API_BASE}/pairs/thumbnail?aoi=${encodeURIComponent(selectedPair.aoi)}&kind=post&name=${encodeURIComponent(selectedPost)}`}
                          alt="Post-event preview"
                        />
                        <figcaption><span className="tag post">POST</span>{selectedPost}</figcaption>
                      </figure>
                    </div>

                    {selectedPair.post_images.length > 1 && (
                      <select className="field" value={selectedPost} onChange={(e) => setSelectedPost(e.target.value)}>
                        {selectedPair.post_images.map((name, i) => {
                          const shift = selectedPair.post_shift_px?.[i];
                          return (
                            <option key={name} value={name}>
                              {name}
                              {shift != null ? ` — ${shift} px offset` : ''}
                              {i === 0 && shift != null ? ' (best aligned)' : ''}
                            </option>
                          );
                        })}
                      </select>
                    )}
                    <div className="note ok">Post image matched automatically from the mapping CSV.</div>
                  </div>
                )}
              </div>
            ) : (
              <div className="stack">
                <div className="upload-row">
                  <label className="file-box">
                    <input type="file" accept=".tif,.tiff" hidden onChange={(e) => handlePreFileChange(e.target.files[0])} />
                    <span className="tag pre">PRE</span>
                    <span className="file-name">{preFile ? preFile.name : 'Select pre-event .tif'}</span>
                  </label>
                  <label className="file-box">
                    <input type="file" accept=".tif,.tiff" hidden onChange={(e) => handlePostFileChange(e.target.files[0])} />
                    <span className="tag post">POST</span>
                    <span className="file-name">{postFile ? postFile.name : 'Matched from pre'}</span>
                  </label>
                </div>

                {autoMatch?.status === 'loading' && <div className="note">Looking up the mapped post image…</div>}
                {autoMatch?.status === 'matched' && (
                  <div className="note ok">
                    Post image matched from the mapping CSV ({aoiName(autoMatch.aoi)}
                    {autoMatch.shift != null ? `, ${autoMatch.shift} px offset` : ''}). Click POST to override.
                  </div>
                )}
                {autoMatch?.status === 'manual' && <div className="note">Using the manually selected post image.</div>}
                {autoMatch?.status === 'none' && (
                  <div className="note warn">This pre image is not in the mapping CSVs — choose the post image manually.</div>
                )}
                {autoMatch?.status === 'error' && (
                  <div className="note warn">Could not reach the pair index — choose the post image manually.</div>
                )}
              </div>
            )}

            <button className="btn-primary" onClick={handlePredict} disabled={loading || !canPredict}>
              {loading ? 'Running inference…' : 'Map flood damage'}
            </button>
          </section>
        </aside>

        {/* ── Middle: before / after ── */}
        <main className="viewport">
          <div className="viewport-bar">
            {result && (
              <div className="layer-bar">
                <span className="layer-label">Layers</span>
                <LayerChip
                  active={layers.buildings}
                  onClick={() => toggle('buildings')}
                  swatch={COLORS.building}
                  title="Predicted building footprints; red = predicted flooded"
                >
                  Buildings
                </LayerChip>
                <LayerChip
                  active={layers.roads}
                  onClick={() => toggle('roads')}
                  swatch={COLORS.road}
                  title="Predicted road centerlines; orange = predicted flooded"
                >
                  Roads
                </LayerChip>
                <LayerChip
                  active={layers.mask}
                  onClick={() => toggle('mask')}
                  swatch="#22d3ee"
                  title="Raw per-pixel model output, before it is turned into building outlines and road lines"
                >
                  Pixel mask
                </LayerChip>
                {gt && (
                  <LayerChip
                    active={layers.groundTruth}
                    onClick={() => toggle('groundTruth')}
                    swatch={COLORS.groundTruth}
                    title="SpaceNet 8 human annotations (dashed) — what the model is scored against"
                  >
                    Reference labels
                  </LayerChip>
                )}
                {layers.mask && (
                  <label className="slider">
                    Opacity
                    <input
                      type="range"
                      min="0.1"
                      max="1"
                      step="0.05"
                      value={layers.opacity}
                      onChange={(e) => setLayers((l) => ({ ...l, opacity: parseFloat(e.target.value) }))}
                    />
                  </label>
                )}
              </div>
            )}

            <div className="zoom-controls">
              <button onClick={() => zoomCenter(1.25)} aria-label="Zoom in" disabled={!result}>+</button>
              <button onClick={() => zoomCenter(0.8)} aria-label="Zoom out" disabled={!result}>−</button>
              <button onClick={() => setView(INITIAL_VIEW)} disabled={!result}>Reset</button>
            </div>
          </div>

          {!result ? (
            <div className="empty-state">
              <div className="empty-icon">🛰</div>
              <h3>Select an image pair to begin</h3>
              <p>
                Choose a pre-event tile on the left — its post-event image is matched automatically from the
                dataset mapping — then run the model to map flooded buildings and roads.
              </p>
            </div>
          ) : (
            <div className="stage">
              <div className="compare-grid">
                <div className="compare-pane">
                  <div className="pane-head"><span className="tag pre">PRE</span> Before the event</div>
                  <div className="canvas-stage">
                    <canvas ref={preCanvasRef} className="scene" {...canvasProps} />
                  </div>
                </div>
                <div className="compare-pane">
                  <div className="pane-head"><span className="tag post">POST</span> After the event</div>
                  <div className="canvas-stage">
                    <canvas ref={postCanvasRef} className="scene" {...canvasProps} />
                  </div>
                </div>
              </div>

              <div className="legend">
                <span><i style={{ background: COLORS.building }} /> Building</span>
                <span><i style={{ background: COLORS.floodedBuilding }} /> Flooded building</span>
                <span><i className="line" style={{ background: COLORS.road }} /> Passable road</span>
                <span><i className="line" style={{ background: COLORS.floodedRoad }} /> Flooded road</span>
                {gt && layers.groundTruth && <span><i className="dashed" /> Reference labels (human annotation)</span>}
                {layers.mask && <span><i style={{ background: '#22d3ee' }} /> Pixel mask (raw model output)</span>}
              </div>
            </div>
          )}
        </main>

        {/* ── Right: results ── */}
        <aside className="panel panel-results">
          {!summary ? (
            <section className="card placeholder">
              <div className="card-head">
                <span className="card-tag">RESULT</span>
                <h2 className="card-title">Damage summary</h2>
              </div>
              <p className="placeholder-text">
                {loading ? 'Running the model on the selected pair…' : 'Results appear here after you map an image pair.'}
              </p>
            </section>
          ) : (
            <>
              <section className="card">
                <div className="card-head">
                  <span className="card-tag">RESULT</span>
                  <h2 className="card-title">Damage summary</h2>
                </div>

                <div className="stat-grid">
                  <StatTile label="Buildings" value={summary.buildings} />
                  <StatTile
                    label="Flooded buildings"
                    value={summary.flooded_buildings}
                    sub={`${summary.flooded_building_pct}% of buildings`}
                    tone="danger"
                  />
                  <StatTile label="Road network" value={`${summary.road_km.toFixed(2)} km`} />
                  <StatTile
                    label="Flooded roads"
                    value={`${summary.flooded_road_km.toFixed(2)} km`}
                    sub={`${summary.flooded_road_pct}% of road length`}
                    tone="warn"
                  />
                </div>

                <div className="meta-line">
                  GSD {summary.gsd_m} m/px
                  {summary.inference_s != null && ` · ${summary.inference_s} s`}
                  {result.pair && ` · ${result.pair.label.replace('.geojson', '')}`}
                </div>

                {result.alignment && (
                  <div className={`note ${result.alignment.applied ? 'ok' : ''}`}>
                    {result.alignment.applied
                      ? `Co-registered: residual offset of ${result.alignment.residual_shift_px} px corrected after resampling to the pre-event grid.`
                      : `Post image resampled to the pre-event grid${result.alignment.residual_shift_px == null ? '; residual offset not estimable.' : '.'}`}
                  </div>
                )}
                {result.model?.stub && <div className="note warn">Stub mode: no model loaded, the masks are synthetic.</div>}
              </section>

              {gt && (
                <section className="card">
                  <div className="card-head">
                    <span className="card-tag">EVALUATION</span>
                    <h2 className="card-title">Accuracy vs ground truth</h2>
                  </div>

                  {gt.metrics ? (
                    <table className="metric-table">
                      <thead>
                        <tr>
                          <th>Class</th>
                          <th>IoU %</th>
                          <th>F1 %</th>
                        </tr>
                      </thead>
                      <tbody>
                        {ACCURACY_ROWS.map(([key, label]) => (
                          <tr key={key}>
                            <td>{label}</td>
                            <td>{pct(gt.metrics[`iou_${key}`])}</td>
                            <td>{pct(gt.metrics[`f1_${key}`])}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ) : (
                    <div className="note">Pixel metrics unavailable for this prediction size.</div>
                  )}

                  <div className="meta-line">
                    Reference: {gt.summary.buildings} buildings ({gt.summary.flooded_buildings} flooded) ·{' '}
                    {gt.summary.road_km.toFixed(2)} km roads ({gt.summary.flooded_road_km.toFixed(2)} km flooded)
                  </div>
                </section>
              )}
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
