import Head from 'next/head';
import Link from 'next/link';
import { useAuth } from '@/context/authContext';
import { Loader } from 'lucide-react';
import styles from '@/styles/Home.module.css';

export default function Help() {
    const { isAuthenticated, isLoading, logout, loginWithRedirect } = useAuth();
    return (
        <div className={styles.container}>
            <Head>
                <title>Help | Flood Risk Analysis</title>
            </Head>

            <header className={styles.header}>
                <Link href="/" className={styles.brand} style={{ textDecoration: 'none' }}>Flood Risk Analysis</Link>
                <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    <Link href="/api" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>API</Link>
                    <Link href="/about" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>About</Link>
                    <Link href="/help" style={{ color: 'var(--text-main)', fontSize: '0.9rem', textDecoration: 'none', fontWeight: 600 }}>Help</Link>
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
                <h1 className={styles.title} style={{ fontSize: '2.5rem', marginBottom: '2rem' }}>Understanding Risk Metrics</h1>

                <section style={{ marginBottom: '3rem' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--text-main)' }}>What does the Risk Percentage mean?</h2>
                    <p style={{ lineHeight: '1.6', color: 'var(--text-dim)', marginBottom: '1rem' }}>
                        The Flood Risk Percentage (0-100%) represents the estimated probability and severity of flooding for a specific area.
                    </p>
                    <p style={{ lineHeight: '1.6', color: 'var(--text-dim)' }}>
                        This metric is calculated using a multimodal machine learning model that analyzes:
                    </p>
                    <ul style={{ listStyle: 'disc', marginLeft: '1.5rem', marginTop: '1rem', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                        <li><strong>Satellite Imagery:</strong> Identifying land cover types (e.g., water, buildings, vegetation).</li>
                        <li><strong>Depth/Elevation Maps:</strong> Assessing terrain features that contribute to water accumulation.</li>
                    </ul>
                </section>

                <section style={{ marginBottom: '3rem' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--text-main)' }}>Interpreting the Data</h2>
                    <div style={{ display: 'grid', gap: '1rem' }}>
                        <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
                            <h3 style={{ marginBottom: '0.5rem', color: '#059669' }}>Low Risk (0 - 30%)</h3>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>Areas with high elevation or permeable surfaces that are unlikely to experience significant flooding.</p>
                        </div>
                        <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
                            <h3 style={{ marginBottom: '0.5rem', color: '#d97706' }}>Moderate Risk (30 - 70%)</h3>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>Areas that may experience flooding during severe weather events or are in proximity to water bodies.</p>
                        </div>
                        <div style={{ padding: '1.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
                            <h3 style={{ marginBottom: '0.5rem', color: '#dc2626' }}>High Risk (70 - 100%)</h3>
                            <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>Critical zones such as existing water bodies, low-lying depressions, or impermeable surfaces in flood-prone regions.</p>
                        </div>
                    </div>
                </section>
            </main>
        </div>
    );
}
