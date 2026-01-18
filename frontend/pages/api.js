import Head from 'next/head';
import Link from 'next/link';
import { useAuth } from '@/context/authContext';
import { Loader } from 'lucide-react';
import styles from '@/styles/Home.module.css';

export default function APIDocs() {
    const { isAuthenticated, isLoading, logout, loginWithRedirect } = useAuth();
    return (
        <div className={styles.container}>
            <Head>
                <title>API Docs | Flood Risk Analysis</title>
            </Head>

            <header className={styles.header}>
                <Link href="/" className={styles.brand} style={{ textDecoration: 'none' }}>Flood Risk Analysis</Link>
                <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    <Link href="/api" style={{ color: 'var(--text-main)', fontSize: '0.9rem', textDecoration: 'none', fontWeight: 600 }}>API</Link>
                    <Link href="/about" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>About</Link>
                    <Link href="/help" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>Help</Link>
                    <a href="https://github.com/danielrhee/MultimodalFloodRiskAnalysis" target="_blank" rel="noopener" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>GitHub</a>
                    {isLoading ? (
                        <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} />
                    ) : isAuthenticated ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                            <button
                                onClick={logout}
                                style={{
                                    background: 'none', border: 'none', padding: 0, fontSize: '0.85rem', cursor: 'pointer',
                                    color: 'var(--text-dim)'
                                }}
                            >
                                Logout
                            </button>
                            <Link
                                href="/portal?type=person"
                                style={{
                                    display: 'flex', alignItems: 'center', gap: '0.25rem',
                                    background: '#000', color: '#fff', border: 'none', borderRadius: '4px',
                                    padding: '0.4rem 0.75rem', fontSize: '0.85rem', cursor: 'pointer',
                                    textDecoration: 'none'
                                }}
                            >
                                Portal
                            </Link>
                        </div>
                    ) : (
                        <button
                            onClick={() => loginWithRedirect()}
                            style={{
                                display: 'flex', alignItems: 'center', gap: '0.25rem',
                                background: '#000', color: '#fff', border: 'none', borderRadius: '4px',
                                padding: '0.4rem 0.75rem', fontSize: '0.85rem', cursor: 'pointer'
                            }}
                        >
                            Sign In
                        </button>
                    )}
                </nav>
            </header>

            <main className={styles.main} style={{ alignItems: 'flex-start', maxWidth: '800px' }}>
                <h1 className={styles.title} style={{ fontSize: '2.5rem', marginBottom: '2rem' }}>Developer API</h1>

                <p style={{ color: 'var(--text-dim)', marginBottom: '2rem', fontSize: '1.1rem', lineHeight: '1.6' }}>
                    Integrate our multimodal flood risk assessment model directly into your urban planning or insurance workflows.
                </p>

                <section style={{ marginBottom: '3rem', width: '100%' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--text-main)' }}>Authentication</h2>
                    <p style={{ color: 'var(--text-dim)', marginBottom: '1rem' }}>
                        All requests must include a Bearer token in the header.
                    </p>
                    <pre style={{ background: '#1e1e1e', color: '#fff', padding: '1rem', borderRadius: '8px', fontSize: '0.9rem' }}>
                        Authorization: Bearer YOUR_API_KEY
                    </pre>
                </section>

                <section style={{ marginBottom: '3rem', width: '100%' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--text-main)' }}>Endpoints</h2>

                    <div style={{ display: 'grid', gap: '1.5rem' }}>
                        <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                <span style={{ background: '#10b981', color: '#fff', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold' }}>POST</span>
                                <code style={{ fontWeight: 'bold' }}>/analyze</code>
                            </div>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '1rem' }}>
                                Upload images to get a risk heatmap and land classification.
                            </p>
                            <h4 style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>Example cURL:</h4>
                            <pre style={{ background: '#1e1e1e', color: '#fff', padding: '1rem', borderRadius: '4px', fontSize: '0.8rem', overflowX: 'auto' }}>
                                {`curl -X POST https://api.floodrisk.io/analyze \\
  -H "Authorization: Bearer \${KEY}" \\
  -F "image=@satellite.png" \\
  -F "depthMap=@elevation.tif"`}
                            </pre>
                        </div>

                        <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                                <span style={{ background: '#3b82f6', color: '#fff', padding: '0.2rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: 'bold' }}>POST</span>
                                <code style={{ fontWeight: 'bold' }}>/chat</code>
                            </div>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '1rem' }}>
                                Query our AI model about flood mitigation strategies for a specific region.
                            </p>
                            <h4 style={{ fontSize: '0.8rem', color: 'var(--text-dim)', marginBottom: '0.5rem' }}>Example Request:</h4>
                            <pre style={{ background: '#1e1e1e', color: '#fff', padding: '1rem', borderRadius: '4px', fontSize: '0.8rem', overflowX: 'auto' }}>
                                {`{
  "message": "How can I reduce flood risk in this wetland?",
  "projectId": "65a..."
}`}
                            </pre>
                        </div>
                    </div>
                </section>
            </main>

            <style jsx global>{`
                pre { margin: 0; }
                code { font-family: 'ui-monospace', monospace; }
            `}</style>
        </div>
    );
}
