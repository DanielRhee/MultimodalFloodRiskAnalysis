import Head from 'next/head';
import Link from 'next/link';
import { useAuth } from '@/context/authContext';
import { Loader } from 'lucide-react';
import styles from '@/styles/Home.module.css';

export default function About() {
    const { isAuthenticated, isLoading, logout, loginWithRedirect } = useAuth();

    return (
        <div className={styles.container}>
            <Head>
                <title>About | Flood Risk Analysis</title>
            </Head>

            <header className={styles.header}>
                <Link href="/" className={styles.brand} style={{ textDecoration: 'none' }}>Flood Risk Analysis</Link>
                <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                    <Link href="/api" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>API</Link>
                    <Link href="/about" style={{ color: 'var(--text-main)', fontSize: '0.9rem', textDecoration: 'none', fontWeight: 600 }}>About</Link>
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

            <main className={styles.main} style={{ alignItems: 'flex-start', maxWidth: '900px' }}>
                <h1 className={styles.title} style={{ fontSize: '3rem', marginBottom: '1.5rem', letterSpacing: '-0.04em' }}>
                    Preventing Deadly, Costly, and Unsustainable Flooding.
                </h1>

                <section style={{ marginBottom: '3.5rem' }}>
                    <p style={{ lineHeight: '1.7', color: 'var(--text-main)', marginBottom: '1.5rem', fontSize: '1.2rem', fontWeight: 500 }}>
                        Flood Risk Analysis emerged from a high-intensity 48-hour hackathon, driven by a single urgent challenge:
                        helping communities adapt to a changing climate by learning to live with water rather than fighting against it.
                    </p>
                    <p style={{ lineHeight: '1.7', color: 'var(--text-dim)', fontSize: '1.1rem' }}>
                        Our AI-powered platform bridges the gap between complex spatial data and human action. By combining
                        high-resolution satellite imagery analysis with advanced elevation modeling, we deliver precise,
                        actionable flood risk intelligence to everyone—from individual homeowners checking their backyards
                        to city planners shaping the neighborhoods of tomorrow.
                    </p>
                </section>

                <section style={{ marginBottom: '3.5rem', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                    <div style={{ padding: '2rem', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
                        <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: 'var(--text-main)' }}>For Individuals</h3>
                        <p style={{ fontSize: '0.95rem', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                            We allow easy acces critical environmental data through a consumer chatbot.
                            It translates thousands of data points into simple conversations, helping you
                            understand the specific risks in your neighborhood and how to mitigate them.
                        </p>
                    </div>
                    <div style={{ padding: '2rem', background: 'var(--surface)', borderRadius: '12px', border: '1px solid var(--border)' }}>
                        <h3 style={{ fontSize: '1.25rem', marginBottom: '1rem', color: 'var(--text-main)' }}>For Professionals</h3>
                        <p style={{ fontSize: '0.95rem', color: 'var(--text-dim)', lineHeight: '1.6' }}>
                            Our enterprise tools empower municipalities and developers to analyze entire regions.
                            With polygon selection, collaborative annotation, and pixel level terrain classification,
                            teams can make development decisions that preserve natural watersheds.
                        </p>
                    </div>
                </section>

                <section style={{ marginBottom: '3.5rem' }}>
                    <h2 style={{ fontSize: '1.75rem', marginBottom: '1.25rem', color: 'var(--text-main)' }}>The Intelligence Layer</h2>
                    <p style={{ lineHeight: '1.7', color: 'var(--text-dim)', marginBottom: '1.5rem', fontSize: '1.1rem' }}>
                        Our custom high accuracy semantic segmentation model classifies every pixel of terrain discriminating between
                        vegetation, permeable soil, water bodies, and buildings. This granular data feeds into our
                        foundational risk model to generate tile-by-tile assessments.
                    </p>
                    <p style={{ lineHeight: '1.7', color: 'var(--text-dim)', fontSize: '1.1rem' }}>
                        By identifying high risk areas before development occurs, we prevent construction waste and the
                        carbon-intensive cycle of reconstruction. We highlight the natural wetlands and floodplains
                        that serve as our planet's best climate adaptation infrastructure, turning expensive consulting
                        knowledge into universal resilience intelligence.
                    </p>
                </section>

                <section style={{ marginBottom: '4rem', padding: '2rem', borderLeft: '4px solid #000', background: '#f9f9f9' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '1rem', color: 'var(--text-main)' }}>Built to Scale</h2>
                    <p style={{ lineHeight: '1.6', color: 'var(--text-dim)' }}>
                        In just two days, we engineered a complete technical stack optimized for performance and reliability.
                        From computer vision models performing pixel level classification to a production-ready web
                        application secured by Auth0, backed by MongoDB persistence, and served through a clean
                        FastAPI architecture we are ready to provide the data foundation for cities to grow in
                        harmony with their watersheds.
                    </p>
                </section>
            </main>

            <style jsx global>{`
                @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            `}</style>
        </div>
    );
}
