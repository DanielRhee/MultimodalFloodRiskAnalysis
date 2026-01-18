import { useState, useRef } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { Upload, ArrowLeft, Loader, MousePointer, Info } from 'lucide-react';
import styles from '@/styles/Portal.module.css';

export default function Portal() {
    const [satelliteImage, setSatelliteImage] = useState(null);
    const [satellitePreview, setSatellitePreview] = useState(null);
    const [depthMap, setDepthMap] = useState(null);
    const [depthPreview, setDepthPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [clickedRisk, setClickedRisk] = useState(null);

    const satelliteInputRef = useRef(null);
    const depthInputRef = useRef(null);
    const riskMapRef = useRef(null);

    const LAND_CLASSES = [
        { name: "Building", color: "rgb(219, 94, 86)" },
        { name: "Greenhouse", color: "rgb(219, 159, 94)" },
        { name: "Swimming Pool", color: "rgb(0, 191, 255)" },
        { name: "Impervious Surface", color: "rgb(192, 192, 192)" },
        { name: "Pervious Surface", color: "rgb(165, 124, 82)" },
        { name: "Bare Soil", color: "rgb(210, 180, 140)" },
        { name: "Water", color: "rgb(0, 0, 255)" },
        { name: "Herbaceous Vegetation", color: "rgb(255, 255, 255)" },
        { name: "Snow", color: "rgb(144, 238, 144)" },
        { name: "Agricultural Land", color: "rgb(255, 255, 0)" },
        { name: "Plowed Land", color: "rgb(139, 69, 19)" },
        { name: "Vineyard", color: "rgb(128, 0, 128)" },
        { name: "Deciduous", color: "rgb(34, 139, 34)" },
        { name: "Coniferous", color: "rgb(0, 100, 0)" },
        { name: "Brushwood", color: "rgb(154, 205, 50)" },
        { name: "Clear Cut", color: "rgb(64, 64, 64)" },
        { name: "Ligneous", color: "rgb(96, 96, 96)" },
        { name: "Mixed", color: "rgb(128, 128, 128)" },
        { name: "Undefined", color: "rgb(0, 0, 0)" }
    ];

    const handleFile = (e, type) => {
        const file = e.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            if (type === 'satellite') {
                setSatelliteImage(file);
                setSatellitePreview(url);
            } else {
                setDepthMap(file);
                setDepthPreview(url);
            }
        }
    };

    const analyzeRisk = async () => {
        if (!satelliteImage) return;

        setIsAnalyzing(true);
        setError(null);
        setResults(null);
        setClickedRisk(null);

        const formData = new FormData();
        formData.append('image', satelliteImage);
        if (depthMap) {
            formData.append('depthMap', depthMap);
            formData.append('depthMin', '0');
            formData.append('depthMax', '10');
        }

        try {
            const response = await fetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Analysis failed');
            const data = await response.json();
            setResults(data);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleMapClick = async (e) => {
        if (!results || !riskMapRef.current) return;

        const img = riskMapRef.current;

        // Get click coordinates relative to the image element
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Calculate percentage coordinates (0.0 to 1.0)
        const pctX = Math.max(0, Math.min(1, x / rect.width));
        const pctY = Math.max(0, Math.min(1, y / rect.height));

        try {
            const response = await fetch(`http://localhost:8000/analyze/${results.analysisId}/risk-point?pctX=${pctX}&pctY=${pctY}`);
            if (!response.ok) throw new Error('Failed to fetch risk');
            const data = await response.json();
            setClickedRisk(data.risk);
        } catch (err) {
            console.error(err);
        }
    };

    return (
        <div className={styles.container}>
            <Head>
                <title>Portal | Flood Risk Analysis</title>
            </Head>

            <header className={styles.header}>
                <div className={styles.brand}> Flood Risk Analysis Consumer</div>
                <Link href="/" className={styles.navLink}>
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <ArrowLeft size={16} /> Back
                    </span>
                </Link>
            </header>

            <main className={styles.main}>
                {/* Left Sidebar: Input */}
                <div className={styles.sidebar}>
                    <div className={styles.card}>
                        <div className={styles.cardHeader}>Data Input</div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <span className={styles.label}>Satellite Image *</span>
                            <div className={styles.uploadZone} onClick={() => satelliteInputRef.current?.click()}>
                                <input type="file" hidden ref={satelliteInputRef} accept="image/*" onChange={(e) => handleFile(e, 'satellite')} />
                                {satellitePreview ? (
                                    <div className={styles.preview}><img src={satellitePreview} alt="Sat" /></div>
                                ) : (
                                    <div><Upload size={20} style={{ marginBottom: '0.5rem' }} /><br />Upload JPG/PNG</div>
                                )}
                            </div>
                        </div>

                        <div>
                            <span className={styles.label}>Depth Map (Optional)</span>
                            <div className={styles.uploadZone} onClick={() => depthInputRef.current?.click()}>
                                <input type="file" hidden ref={depthInputRef} accept="image/*" onChange={(e) => handleFile(e, 'depth')} />
                                {depthPreview ? (
                                    <div className={styles.preview}><img src={depthPreview} alt="Depth" /></div>
                                ) : (
                                    <div><Upload size={20} style={{ marginBottom: '0.5rem' }} /><br />Upload Map</div>
                                )}
                            </div>
                        </div>

                        <button
                            className={styles.analyzeBtn}
                            onClick={analyzeRisk}
                            disabled={!satelliteImage || isAnalyzing}
                        >
                            {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
                        </button>

                        {error && <div style={{ color: 'red', marginTop: '1rem', fontSize: '0.9rem' }}>{error}</div>}
                    </div>
                </div>

                {/* Center: Content */}
                <div className={styles.content}>
                    {isAnalyzing && (
                        <div className={styles.loader}>
                            <Loader className="animate-spin" size={24} style={{ marginRight: '0.5rem', animation: 'spin 1s linear infinite' }} />
                            Processing Model...
                        </div>
                    )}

                    {!results && !isAnalyzing && (
                        <div className={styles.loader} style={{ flexDirection: 'column' }}>
                            Waiting for input...
                        </div>
                    )}

                    {results && (
                        <div className={styles.resultsGrid}>
                            <div className={styles.resultCard}>
                                <div className={styles.resultHeader}>
                                    Risk Heatmap
                                    <span style={{ fontSize: '0.75rem', fontWeight: 400, marginLeft: '0.5rem', color: '#64748b' }}>(Interactive)</span>
                                </div>
                                <div className={styles.resultImage} style={{ cursor: 'crosshair', position: 'relative' }}>
                                    <img
                                        ref={riskMapRef}
                                        src={`http://localhost:8000/analyze/${results.analysisId}/risk-map`}
                                        alt="Risk"
                                        onClick={handleMapClick}
                                        style={{ display: 'block' }}
                                    />
                                </div>
                            </div>

                            <div className={styles.resultCard}>
                                <div className={styles.resultHeader}>Land Classification</div>
                                <div className={styles.resultImage}>
                                    <img src={`http://localhost:8000/analyze/${results.analysisId}/land-classification`} alt="Land Class" />
                                </div>
                                <div style={{ padding: '1rem', borderTop: '1px solid var(--border)' }}>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Legend</div>
                                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))', gap: '0.5rem' }}>
                                        {LAND_CLASSES.map((cls) => (
                                            <div key={cls.name} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                                                <div style={{ width: '12px', height: '12px', borderRadius: '2px', backgroundColor: cls.color, border: '1px solid rgba(0,0,0,0.1)' }}></div>
                                                <span>{cls.name}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* Right Sidebar: Statistics */}
                <div className={styles.rightSidebar}>
                    {results ? (
                        <div className={styles.card}>
                            <div className={styles.cardHeader}>Risk Analysis</div>

                            <div className={styles.statItem} style={{ marginBottom: '1.5rem' }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <span className={styles.label} style={{ marginBottom: 0 }}>Overall Average Risk</span>
                                    <Link href="/help" target="_blank">
                                        <Info size={14} color="var(--text-dim)" style={{ cursor: 'pointer' }} />
                                    </Link>
                                </div>
                                <span className={styles.statVal} style={{ color: results.averageRisk > 50 ? '#dc2626' : '#059669', fontSize: '2rem' }}>
                                    {(results.averageRisk).toFixed(1)}%
                                </span>
                            </div>

                            <div className={styles.statItem} style={{ borderTop: '1px solid #e5e5e5', paddingTop: '1.5rem', marginBottom: '1rem' }}>
                                <span className={styles.label}>Selected Area Risk</span>
                                {clickedRisk !== null ? (
                                    <span className={styles.statVal} style={{ fontSize: '2rem' }}>
                                        {(clickedRisk).toFixed(1)}%
                                    </span>
                                ) : (
                                    <span style={{ color: '#999', fontSize: '0.9rem', fontStyle: 'italic' }}>
                                        Click map to inspect
                                    </span>
                                )}
                            </div>

                            <div style={{ backgroundColor: '#f8fafc', padding: '1rem', borderRadius: '4px', fontSize: '0.85rem', color: '#64748b', display: 'flex', gap: '0.5rem' }}>
                                <Info size={16} />
                                <span>Risk values (0-100%) represent the probability of flooding based on terrain and coverage.</span>
                            </div>
                        </div>
                    ) : (
                        <div className={styles.card} style={{ opacity: 0.5 }}>
                            <div className={styles.cardHeader}>Risk Analysis</div>
                            <p style={{ fontSize: '0.9rem', color: '#666' }}>Results will appear here...</p>
                        </div>
                    )}
                </div>
            </main>
            <style jsx global>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
        </div>
    );
}
