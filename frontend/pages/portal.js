import { useState, useRef, useEffect, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { useAuth } from '@/context/authContext';
import { Upload, ArrowLeft, Loader, Info, MessageSquare, Send, MapPin, Pentagon, Trash2, Check, Save, FolderOpen, Plus, X, Edit2, LogOut } from 'lucide-react';
import styles from '@/styles/Portal.module.css';

export default function Portal() {
    const router = useRouter();
    const { isAuthenticated, isLoading: authLoading, user, loginWithRedirect, logout, authenticatedFetch, getAccessToken } = useAuth();

    const isConsumer = router.query.type === 'person';
    const isEnterprise = !isConsumer;
    const portalType = isConsumer ? 'consumer' : 'enterprise';

    const [satelliteImage, setSatelliteImage] = useState(null);
    const [satellitePreview, setSatellitePreview] = useState(null);
    const [depthMap, setDepthMap] = useState(null);
    const [depthPreview, setDepthPreview] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [clickedRisk, setClickedRisk] = useState(null);

    const [isChatOpen, setIsChatOpen] = useState(false);
    const [chatMessages, setChatMessages] = useState([]);
    const [chatInput, setChatInput] = useState("");
    const [isChatLoading, setIsChatLoading] = useState(false);

    const [annotations, setAnnotations] = useState([]);
    const [activeMode, setActiveMode] = useState(null);
    const [polygonVertices, setPolygonVertices] = useState([]);
    const [polygonRiskResult, setPolygonRiskResult] = useState(null);
    const [editingAnnotationId, setEditingAnnotationId] = useState(null);

    const [showProjectModal, setShowProjectModal] = useState(false);
    const [projects, setProjects] = useState([]);
    const [projectsLoading, setProjectsLoading] = useState(true);
    const [currentProject, setCurrentProject] = useState(null);
    const [saveEnabled, setSaveEnabled] = useState(true);
    const [projectName, setProjectName] = useState("");
    const [isEditingName, setIsEditingName] = useState(false);
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(null);

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

    useEffect(() => {
        if (!authLoading && !isAuthenticated) {
            loginWithRedirect({ appState: { returnTo: router.asPath } });
        }
    }, [authLoading, isAuthenticated, loginWithRedirect, router.asPath]);

    useEffect(() => {
        if (isAuthenticated && router.query.type) {
            loadProjects();
            if (isConsumer) {
                fetchUserHistory();
            }
            setShowProjectModal(true);
        }
    }, [isAuthenticated, router.query.type, isConsumer]);

    const loadProjects = async () => {
        try {
            setProjectsLoading(true);
            const response = await authenticatedFetch('http://localhost:8000/projects');
            if (response.ok) {
                const data = await response.json();
                setProjects(data.projects.filter(p => p.projectType === portalType));
            }
        } catch (err) {
            console.error("Failed to load projects:", err);
        } finally {
            setProjectsLoading(false);
        }
    };

    const fetchUserHistory = async () => {
        try {
            const response = await authenticatedFetch('http://localhost:8000/me');
            if (response.ok) {
                const data = await response.json();
                if (!currentProject) {
                    setChatMessages(data.chatHistory || []);
                }
            }
        } catch (err) {
            console.error("Failed to load user history:", err);
        }
    };

    const createNewProject = async (name) => {
        try {
            const response = await authenticatedFetch('http://localhost:8000/projects', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, projectType: portalType })
            });
            if (response.ok) {
                const data = await response.json();
                setCurrentProject(data.projectId);
                setProjectName(name);
                setShowProjectModal(false);
                setSaveEnabled(true);
            }
        } catch (err) {
            console.error("Failed to create project:", err);
        }
    };

    const fetchAuthImage = async (projectId, fileId) => {
        try {
            const response = await authenticatedFetch(`http://localhost:8000/projects/${projectId}/files/${fileId}`);
            if (response.ok) {
                const blob = await response.blob();
                return URL.createObjectURL(blob);
            }
        } catch (err) {
            console.error(`Failed to load image ${fileId}:`, err);
        }
        return null;
    };

    const loadProject = async (projectId) => {
        try {
            const response = await authenticatedFetch(`http://localhost:8000/projects/${projectId}`);
            if (response.ok) {
                const data = await response.json();
                setCurrentProject(projectId);
                setProjectName(data.name);
                setAnnotations(data.annotations || []);
                setPolygonVertices(data.polygonVertices || []);
                setChatMessages(data.chatHistory || []);

                // Load images if they exist
                if (data.satelliteFileId) {
                    loadImage(projectId, data.satelliteFileId, 'satellite');
                } else {
                    setSatelliteImage(null);
                    setSatellitePreview(null);
                }

                if (data.depthMapFileId) {
                    loadImage(projectId, data.depthMapFileId, 'depth');
                } else {
                    setDepthMap(null);
                    setDepthPreview(null);
                }

                if (data.analysisMetrics && data.riskMapFileId) {
                    // Fetch images as blobs for authenticated display
                    const riskMapUrl = await fetchAuthImage(projectId, data.riskMapFileId);
                    const landClassUrl = data.landClassificationFileId ? await fetchAuthImage(projectId, data.landClassificationFileId) : null;

                    if (riskMapUrl) {
                        setResults({
                            averageRisk: data.analysisMetrics.averageRisk,
                            riskMapUrl: riskMapUrl,
                            landClassUrl: landClassUrl,
                            riskByLandClass: data.analysisMetrics.riskByLandClass,
                            landClassDistribution: data.analysisMetrics.landClassDistribution,
                            isSaved: true,
                            analysisId: null
                        });
                    }
                } else {
                    setResults(null);
                }

                setSaveEnabled(true);
                setShowProjectModal(false);
            }
        } catch (err) {
            console.error("Failed to load project:", err);
        }
    };

    const loadImage = async (projectId, fileId, type) => {
        try {
            const response = await authenticatedFetch(`http://localhost:8000/projects/${projectId}/files/${fileId}`);
            if (response.ok) {
                const blob = await response.blob();
                const file = new File([blob], `${type}.png`, { type: "image/png" });
                const url = URL.createObjectURL(blob);

                if (type === 'satellite') {
                    setSatelliteImage(file);
                    setSatellitePreview(url);
                } else {
                    setDepthMap(file);
                    setDepthPreview(url);
                }
            }
        } catch (err) {
            console.error(`Failed to load ${type} image:`, err);
        }
    };

    // ... (rest of the code)

    const uploadFile = async (file, type) => {
        if (!currentProject || !saveEnabled) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('fileType', type === 'satellite' ? 'satellite' : 'depthMap');

        try {
            await authenticatedFetch(`http://localhost:8000/projects/${currentProject}/files`, {
                method: 'POST',
                body: formData
            });
        } catch (err) {
            console.error(`Failed to upload ${type}:`, err);
        }
    };

    const handleFile = (e, type) => {
        const file = e.target.files[0];
        if (file) {
            const url = URL.createObjectURL(file);
            if (type === 'satellite') {
                setSatelliteImage(file);
                setSatellitePreview(url);
                uploadFile(file, 'satellite');
            } else {
                setDepthMap(file);
                setDepthPreview(url);
                uploadFile(file, 'depth');
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

        if (currentProject && saveEnabled) {
            formData.append('projectId', currentProject);
            formData.append('saveEnabled', 'true');
        }

        try {
            const response = await authenticatedFetch('http://localhost:8000/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Analysis failed');
            const data = await response.json();
            setResults({
                ...data,
                riskMapUrl: `http://localhost:8000/analyze/${data.analysisId}/risk-map`,
                landClassUrl: `http://localhost:8000/analyze/${data.analysisId}/land-classification`,
                isSaved: false
            });
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }
    };

    const handleMapClick = async (e) => {
        if (!results || !riskMapRef.current) return;

        const img = riskMapRef.current;
        const rect = img.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        const pctX = Math.max(0, Math.min(1, x / rect.width));
        const pctY = Math.max(0, Math.min(1, y / rect.height));

        if (isEnterprise && activeMode === 'annotate') {
            const newAnnotation = {
                id: Date.now(),
                x: pctX,
                y: pctY,
                comment: ''
            };
            setAnnotations([...annotations, newAnnotation]);
            setEditingAnnotationId(newAnnotation.id);
            return;
        }

        if (isEnterprise && activeMode === 'polygon') {
            setPolygonVertices([...polygonVertices, { x: pctX, y: pctY }]);
            return;
        }

        try {
            let url;
            let options = {};

            // Robust check for saved state
            const isSavedProject = results.isSaved || (currentProject && !results.analysisId);

            if (isSavedProject) {
                if (!currentProject) return;
                url = `http://localhost:8000/projects/${currentProject}/risk-point?pctX=${pctX}&pctY=${pctY}`;
                const token = await getAccessToken();
                options = {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                };
            } else {
                url = `http://localhost:8000/analyze/${results.analysisId}/risk-point?pctX=${pctX}&pctY=${pctY}`;
            }

            const response = await fetch(url, options);
            if (!response.ok) throw new Error('Failed to fetch risk');
            const data = await response.json();
            setClickedRisk(data.risk);
        } catch (err) {
            console.error(err);
        }
    };

    const calculatePolygonRisk = async () => {
        if (!results || polygonVertices.length < 3) return;

        try {
            let url;
            let options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ vertices: polygonVertices })
            };

            // Robust check for saved state
            const isSavedProject = results.isSaved || (currentProject && !results.analysisId);

            if (isSavedProject) {
                if (!currentProject) return;
                url = `http://localhost:8000/projects/${currentProject}/polygon-risk`;
                const token = await getAccessToken();
                options.headers['Authorization'] = `Bearer ${token}`;
            } else {
                url = `http://localhost:8000/analyze/${results.analysisId}/polygon-risk`;
            }

            const response = await fetch(url, options);
            if (!response.ok) throw new Error('Failed to calculate polygon risk');
            const data = await response.json();
            setPolygonRiskResult(data);
        } catch (err) {
            console.error(err);
        }
    };

    const updateAnnotationComment = (id, comment) => {
        setAnnotations(annotations.map(a => a.id === id ? { ...a, comment } : a));
    };

    const deleteAnnotation = (id) => {
        setAnnotations(annotations.filter(a => a.id !== id));
        if (editingAnnotationId === id) setEditingAnnotationId(null);
    };

    const clearPolygon = () => {
        setPolygonVertices([]);
        setPolygonRiskResult(null);
    };

    const updateProjectName = async () => {
        if (!currentProject || !projectName.trim()) return;
        try {
            await authenticatedFetch(`http://localhost:8000/projects/${currentProject}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: projectName })
            });
            setIsEditingName(false);
            setProjects(projects.map(p => p._id === currentProject ? { ...p, name: projectName, updatedAt: new Date() } : p));
        } catch (err) {
            console.error("Failed to update project name:", err);
        }
    };

    const saveAnnotations = useCallback(async () => {
        if (!currentProject || !saveEnabled || !isEnterprise) return;
        try {
            await authenticatedFetch(`http://localhost:8000/projects/${currentProject}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    annotations,
                    polygonVertices
                })
            });
        } catch (err) {
            console.error("Failed to save project data:", err);
        }
    }, [currentProject, saveEnabled, isEnterprise, annotations, polygonVertices, authenticatedFetch]);

    useEffect(() => {
        if ((annotations.length > 0 || polygonVertices.length > 0) && currentProject && saveEnabled && isEnterprise) {
            const timeout = setTimeout(saveAnnotations, 1000);
            return () => clearTimeout(timeout);
        }
    }, [annotations, polygonVertices, currentProject, saveEnabled, isEnterprise, saveAnnotations]);

    const handleChatSubmit = async (e) => {
        e.preventDefault();
        if (!chatInput.trim()) return;

        const userMsg = chatInput;
        const newHistory = [...chatMessages, { role: 'user', content: userMsg }];
        setChatMessages(newHistory);
        setChatInput("");
        setIsChatLoading(true);

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: userMsg,
                    history: newHistory,
                    projectId: currentProject,
                    saveEnabled: saveEnabled && isConsumer
                })
            });
            const data = await response.json();
            const updatedHistory = [...newHistory, { role: 'model', content: data.response }];
            setChatMessages(updatedHistory);
        } catch (err) {
            console.error(err);
            setChatMessages([...newHistory, { role: 'model', content: "Sorry, I encountered an error. Please try again." }]);
        } finally {
            setIsChatLoading(false);
        }
    };

    const deleteProjectHandler = async (projectId) => {
        try {
            const response = await authenticatedFetch(`http://localhost:8000/projects/${projectId}`, {
                method: 'DELETE'
            });
            if (response.ok) {
                setProjects(projects.filter(p => p._id !== projectId));
                if (currentProject === projectId) {
                    setCurrentProject(null);
                    setSaveEnabled(false);
                    setResults(null);
                    setSatelliteImage(null);
                    setSatellitePreview(null);
                    setDepthMap(null);
                    setDepthPreview(null);
                    setAnnotations([]);
                    setChatMessages([]);
                }
            }
        } catch (err) {
            console.error("Failed to delete project:", err);
        }
    };

    const continueWithoutSaving = () => {
        if (isConsumer) {
            setSaveEnabled(true);
            fetchUserHistory();
        } else {
            setSaveEnabled(false);
        }
        setCurrentProject(null);
        setShowProjectModal(false);
    };

    if (authLoading) {
        return (
            <div className={styles.container}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: '0.5rem' }}>
                    <Loader size={24} style={{ animation: 'spin 1s linear infinite' }} />
                    <span>Loading...</span>
                </div>
                <style jsx global>{`
                    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                `}</style>
            </div>
        );
    }

    if (!isAuthenticated) {
        return null;
    }

    return (
        <div className={styles.container}>
            <Head>
                <title>Portal | Flood Risk Analysis</title>
            </Head>

            <header className={styles.header}>
                <div className={styles.brand}>
                    <Link href="/" style={{ textDecoration: 'none', color: 'inherit' }}>
                        Flood Risk Analysis {isEnterprise ? 'Enterprise' : 'Consumer'}
                    </Link>
                    {currentProject && (
                        <span style={{ marginLeft: '1rem', color: 'var(--text-dim)', fontSize: '0.9rem' }}>
                            {isEditingName ? (
                                <input
                                    type="text"
                                    value={projectName}
                                    onChange={(e) => setProjectName(e.target.value)}
                                    onBlur={updateProjectName}
                                    onKeyDown={(e) => e.key === 'Enter' && updateProjectName()}
                                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.9rem', border: '1px solid var(--border)', borderRadius: '4px' }}
                                    autoFocus
                                />
                            ) : (
                                <>
                                    — {projectName}
                                    <Edit2
                                        size={12}
                                        style={{ marginLeft: '0.5rem', cursor: 'pointer' }}
                                        onClick={() => setIsEditingName(true)}
                                    />
                                </>
                            )}
                        </span>
                    )}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                    {currentProject && (
                        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-dim)' }}>
                            <input
                                type="checkbox"
                                checked={saveEnabled}
                                onChange={(e) => setSaveEnabled(e.target.checked)}
                            />
                            Auto-save
                        </label>
                    )}
                    <button
                        onClick={() => { loadProjects(); setShowProjectModal(true); }}
                        className={styles.navLink}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', background: 'none', border: 'none', padding: '0.4rem 0.75rem', cursor: 'pointer' }}
                    >
                        <FolderOpen size={14} /> Projects
                    </button>
                    <Link href="/api" className={styles.navLink} style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>
                        API
                    </Link>
                    <Link href="/about" className={styles.navLink} style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>
                        About
                    </Link>
                    <Link href="/help" className={styles.navLink} style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>
                        Help
                    </Link>
                    <a href="https://github.com/danielrhee/MultimodalFloodRiskAnalysis" target="_blank" rel="noopener" className={styles.navLink} style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>
                        GitHub
                    </a>
                    <button
                        onClick={logout}
                        style={{
                            background: 'none', border: 'none', padding: 0, fontSize: '0.9rem', cursor: 'pointer',
                            color: 'var(--text-dim)', marginLeft: '0.5rem'
                        }}
                    >
                        Logout
                    </button>
                    <Link
                        href="/portal?type=person"
                        style={{
                            display: 'flex', alignItems: 'center', gap: '0.25rem',
                            background: '#000', color: '#fff', border: 'none', borderRadius: '4px',
                            padding: '0.4rem 0.75rem', fontSize: '0.9rem', cursor: 'pointer',
                            textDecoration: 'none'
                        }}
                    >
                        Portal
                    </Link>
                </div>
            </header>

            {showProjectModal && (
                <div className={styles.modalOverlay}>
                    <div className={styles.modal}>
                        <div className={styles.modalHeader}>
                            <h3>Projects</h3>
                            <button onClick={() => setShowProjectModal(false)} className={styles.modalClose}>
                                <X size={18} />
                            </button>
                        </div>
                        <div className={styles.modalBody}>
                            <button
                                onClick={() => {
                                    const name = prompt("Enter project name:", `Project ${new Date().toLocaleDateString()}`);
                                    if (name) createNewProject(name);
                                }}
                                className={styles.newProjectBtn}
                            >
                                <Plus size={18} /> New Project
                            </button>

                            <button
                                onClick={continueWithoutSaving}
                                className={styles.skipSaveBtn}
                            >
                                Continue {isConsumer ? '' : 'without saving'}
                            </button>

                            {projectsLoading ? (
                                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)' }}>
                                    <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                                </div>
                            ) : projects.length === 0 ? (
                                <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)' }}>
                                    No saved projects yet
                                </div>
                            ) : (
                                <div className={styles.projectList}>
                                    {projects.map(project => (
                                        <div key={project._id} className={styles.projectItem}>
                                            <div
                                                className={styles.projectInfo}
                                                onClick={() => loadProject(project._id)}
                                            >
                                                <div className={styles.projectName}>{project.name}</div>
                                                <div className={styles.projectDate}>
                                                    {new Date(project.updatedAt).toLocaleDateString()}
                                                </div>
                                            </div>
                                            <button
                                                onClick={(e) => { e.stopPropagation(); setShowDeleteConfirm(project._id); }}
                                                className={styles.deleteBtn}
                                            >
                                                <Trash2 size={14} />
                                            </button>
                                            {showDeleteConfirm === project._id && (
                                                <div className={styles.deleteConfirm}>
                                                    <span>Delete?</span>
                                                    <button onClick={() => deleteProjectHandler(project._id)}>Yes</button>
                                                    <button onClick={() => setShowDeleteConfirm(null)}>No</button>
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )
            }

            <main className={styles.main}>
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
                                <div className={styles.resultImage} style={{ cursor: activeMode ? 'crosshair' : 'pointer', position: 'relative' }}>
                                    <img
                                        ref={riskMapRef}
                                        src={results.riskMapUrl}
                                        alt="Risk"
                                        onClick={handleMapClick}
                                        style={{ display: 'block', width: '100%' }}
                                    />

                                    {isEnterprise && annotations.map((ann, idx) => (
                                        <div
                                            key={ann.id}
                                            className={styles.annotationMarker}
                                            style={{
                                                left: `${ann.x * 100}%`,
                                                top: `${ann.y * 100}%`,
                                            }}
                                            onClick={(e) => { e.stopPropagation(); setEditingAnnotationId(ann.id); }}
                                        >
                                            <MapPin size={20} fill="#dc2626" color="#fff" />
                                            <span className={styles.annotationNumber}>{idx + 1}</span>
                                        </div>
                                    ))}

                                    {isEnterprise && polygonVertices.length > 0 && (
                                        <svg className={styles.polygonOverlay} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
                                            {polygonVertices.length > 1 && polygonVertices.map((v, idx) => {
                                                if (idx === 0) return null;
                                                const prev = polygonVertices[idx - 1];
                                                return (
                                                    <line
                                                        key={idx}
                                                        x1={`${prev.x * 100}%`}
                                                        y1={`${prev.y * 100}%`}
                                                        x2={`${v.x * 100}%`}
                                                        y2={`${v.y * 100}%`}
                                                        stroke="#3b82f6"
                                                        strokeWidth="2"
                                                    />
                                                );
                                            })}
                                            {polygonVertices.length >= 3 && (
                                                <line
                                                    x1={`${polygonVertices[polygonVertices.length - 1].x * 100}%`}
                                                    y1={`${polygonVertices[polygonVertices.length - 1].y * 100}%`}
                                                    x2={`${polygonVertices[0].x * 100}%`}
                                                    y2={`${polygonVertices[0].y * 100}%`}
                                                    stroke="#3b82f6"
                                                    strokeWidth="2"
                                                    strokeDasharray="4"
                                                />
                                            )}
                                            {polygonVertices.map((v, idx) => (
                                                <circle
                                                    key={idx}
                                                    cx={`${v.x * 100}%`}
                                                    cy={`${v.y * 100}%`}
                                                    r="6"
                                                    fill="#3b82f6"
                                                    stroke="#fff"
                                                    strokeWidth="2"
                                                />
                                            ))}
                                        </svg>
                                    )}
                                </div>
                            </div>

                            <div className={styles.resultCard}>
                                <div className={styles.resultHeader}>Land Classification</div>
                                <div className={styles.resultImage}>
                                    <img src={results.landClassUrl} alt="Land Class" />
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

                    {isConsumer && (
                        <div className={styles.card}>
                            <div className={styles.cardHeader}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <MessageSquare size={18} />
                                    <span>Flood Risk Assistant</span>
                                </div>
                            </div>

                            <div className={styles.chatMessages}>
                                {chatMessages.length === 0 && (
                                    <div style={{ padding: '1rem', color: '#64748b', fontSize: '0.9rem', textAlign: 'center' }}>
                                        👋 Hi! I can help you understand your flood risk analysis. Ask me anything!
                                    </div>
                                )}
                                {chatMessages.map((msg, idx) => (
                                    <div key={idx} className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.botMessage}`}>
                                        {msg.content}
                                    </div>
                                ))}
                                {isChatLoading && (
                                    <div className={styles.botMessage} style={{ fontStyle: 'italic', color: '#64748b' }}>
                                        Thinking...
                                    </div>
                                )}
                            </div>

                            <form onSubmit={handleChatSubmit}>
                                <div className={styles.chatInputArea}>
                                    <input
                                        type="text"
                                        placeholder="Type your question..."
                                        value={chatInput}
                                        onChange={(e) => setChatInput(e.target.value)}
                                        className={styles.chatInput}
                                    />
                                    <button type="submit" className={styles.sendButton} disabled={!chatInput.trim() || isChatLoading}>
                                        <Send size={16} />
                                    </button>
                                </div>
                            </form>
                        </div>
                    )}

                    {isEnterprise && (
                        <div className={styles.card}>
                            <div className={styles.cardHeader}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                    <Pentagon size={18} />
                                    <span>Planning Tools</span>
                                </div>
                            </div>

                            <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem' }}>
                                <button
                                    className={`${styles.toolButton} ${activeMode === 'annotate' ? styles.toolButtonActive : ''}`}
                                    onClick={() => setActiveMode(activeMode === 'annotate' ? null : 'annotate')}
                                    disabled={!results}
                                >
                                    <MapPin size={16} />
                                    <span>Annotate</span>
                                </button>
                                <button
                                    className={`${styles.toolButton} ${activeMode === 'polygon' ? styles.toolButtonActive : ''}`}
                                    onClick={() => { setActiveMode(activeMode === 'polygon' ? null : 'polygon'); clearPolygon(); }}
                                    disabled={!results}
                                >
                                    <Pentagon size={16} />
                                    <span>Polygon</span>
                                </button>
                            </div>

                            {activeMode && (
                                <div style={{ backgroundColor: '#f0f9ff', padding: '0.75rem', borderRadius: '4px', fontSize: '0.85rem', color: '#0369a1', marginBottom: '1rem' }}>
                                    {activeMode === 'annotate' && 'Click on the map to add annotation markers.'}
                                    {activeMode === 'polygon' && 'Click on the map to add polygon vertices. Click "Calculate" when done.'}
                                </div>
                            )}

                            {annotations.length > 0 && (
                                <div style={{ marginBottom: '1rem' }}>
                                    <div style={{ fontSize: '0.85rem', fontWeight: 600, marginBottom: '0.5rem' }}>Annotations</div>
                                    <div style={{ maxHeight: '150px', overflowY: 'auto' }}>
                                        {annotations.map((ann, idx) => (
                                            <div key={ann.id} style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginBottom: '0.5rem', padding: '0.5rem', backgroundColor: '#f5f5f5', borderRadius: '4px' }}>
                                                <span style={{ backgroundColor: '#dc2626', color: '#fff', borderRadius: '50%', width: '20px', height: '20px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.75rem', flexShrink: 0 }}>{idx + 1}</span>
                                                <div style={{ flex: 1 }}>
                                                    {editingAnnotationId === ann.id ? (
                                                        <input
                                                            type="text"
                                                            value={ann.comment}
                                                            onChange={(e) => updateAnnotationComment(ann.id, e.target.value)}
                                                            onBlur={() => setEditingAnnotationId(null)}
                                                            onKeyDown={(e) => e.key === 'Enter' && setEditingAnnotationId(null)}
                                                            placeholder="Add comment..."
                                                            autoFocus
                                                            style={{ width: '100%', padding: '0.25rem', fontSize: '0.85rem', border: '1px solid #ccc', borderRadius: '4px' }}
                                                        />
                                                    ) : (
                                                        <div onClick={() => setEditingAnnotationId(ann.id)} style={{ cursor: 'text', fontSize: '0.85rem', color: ann.comment ? '#333' : '#999' }}>
                                                            {ann.comment || 'Click to add comment...'}
                                                        </div>
                                                    )}
                                                </div>
                                                <button onClick={() => deleteAnnotation(ann.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#666' }}>
                                                    <Trash2 size={14} />
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {activeMode === 'polygon' && (
                                <div style={{ marginBottom: '1rem' }}>
                                    <div style={{ fontSize: '0.85rem', marginBottom: '0.5rem' }}>
                                        Vertices: {polygonVertices.length}
                                    </div>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <button
                                            className={styles.toolButton}
                                            onClick={calculatePolygonRisk}
                                            disabled={polygonVertices.length < 3}
                                            style={{ flex: 1 }}
                                        >
                                            <Check size={16} />
                                            <span>Calculate Risk</span>
                                        </button>
                                        <button
                                            className={styles.toolButton}
                                            onClick={clearPolygon}
                                            disabled={polygonVertices.length === 0}
                                        >
                                            <Trash2 size={16} />
                                        </button>
                                    </div>
                                </div>
                            )}

                            {polygonRiskResult && (
                                <div style={{ backgroundColor: '#f0fdf4', padding: '1rem', borderRadius: '4px', marginTop: '1rem' }}>
                                    <div style={{ fontSize: '0.85rem', color: '#166534', marginBottom: '0.25rem' }}>Polygon Area Risk</div>
                                    <div style={{ fontSize: '1.5rem', fontWeight: 600, color: polygonRiskResult.averageRisk > 50 ? '#dc2626' : '#059669' }}>
                                        {polygonRiskResult.averageRisk.toFixed(1)}%
                                    </div>
                                    <div style={{ fontSize: '0.75rem', color: '#666', marginTop: '0.25rem' }}>
                                        Based on {polygonRiskResult.pixelsAnalyzed.toLocaleString()} pixels
                                    </div>
                                </div>
                            )}

                            {!results && (
                                <div style={{ color: '#999', fontSize: '0.85rem', fontStyle: 'italic' }}>
                                    Run analysis to enable tools
                                </div>
                            )}
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
