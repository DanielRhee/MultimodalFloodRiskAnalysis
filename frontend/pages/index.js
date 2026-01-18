import Head from 'next/head';
import Link from 'next/link';
import { useState } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '@/context/authContext';
import { ChevronDown, User, Briefcase, LogIn, LogOut, Loader } from 'lucide-react';
import styles from '@/styles/Home.module.css';

export default function Home() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const { isAuthenticated, isLoading, user, loginWithRedirect, logout } = useAuth();
  const router = useRouter();

  const handlePortalClick = (type) => {
    if (!isAuthenticated) {
      loginWithRedirect({ appState: { returnTo: `/portal?type=${type}` } });
    } else {
      router.push(`/portal?type=${type}`);
    }
  };

  return (
    <div className={styles.container}>
      <Head>
        <title>Flood Risk Analytics</title>
        <meta name="description" content="Minimal Flood Risk Analysis Tool" />
      </Head>

      <header className={styles.header}>
        <Link href="/" className={styles.brand} style={{ textDecoration: 'none' }}>Flood Risk Analysis</Link>
        <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <Link href="/api" style={{ color: 'var(--text-dim)', fontSize: '0.9rem', textDecoration: 'none' }}>API</Link>
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

      <main className={styles.main}>
        <h1 className={styles.title}>
          Multimodal Flood Risk<br />Analysis Platform
        </h1>

        <p className={styles.subtitle}>
          A powerful and efficient tool combining satellite imagery and depth maps to predict long term flood risks and increase sustainability in urban planning.
        </p>

        <div className={styles.ctaWrapper}>
          <button
            className={styles.ctaButton}
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            Launch Portal
            <ChevronDown size={16} style={{ marginLeft: '0.5rem', transition: 'transform 0.2s', transform: isMenuOpen ? 'rotate(180deg)' : 'rotate(0deg)' }} />
          </button>

          {isMenuOpen && (
            <div className={styles.menu}>
              <div className={styles.menuItem} onClick={() => handlePortalClick('person')} style={{ cursor: 'pointer' }}>
                <div className={styles.menuIcon}>
                  <User size={18} />
                </div>
                <div className={styles.menuContent}>
                  <span className={styles.menuTitle}>Person</span>
                  <span className={styles.menuDesc}>For individual property checks</span>
                </div>
              </div>

              <div className={styles.menuItem} onClick={() => handlePortalClick('planner')} style={{ cursor: 'pointer' }}>
                <div className={styles.menuIcon}>
                  <Briefcase size={18} />
                </div>
                <div className={styles.menuContent}>
                  <span className={styles.menuTitle}>
                    Planner
                    <span className={styles.badge}>Enterprise</span>
                  </span>
                  <span className={styles.menuDesc}>For urban planning & analysis</span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className={styles.features}>
          <div className={styles.feature}>
            <h3>Precise Analysis</h3>
            <p>Utilizes an advanced multimodal foundation model to detect water bodies and elevation risks.</p>
          </div>
          <div className={styles.feature}>
            <h3>Instant Feedback</h3>
            <p>Get immediate risk assessments processed for areas based on the local area</p>
          </div>
          <div className={styles.feature}>
            <h3>Sustainable</h3>
            <p>Provides instant analysis on smart building zones and critical ecosystems</p>
          </div>
        </div>

        <h2 className={styles.sectionTitle}>How to Use</h2>
        <div className={styles.features} style={{ marginTop: '0', borderTop: 'none' }}>
          <div className={styles.feature}>
            <h3>Consumer</h3>
            <p>Choose smart and sustainable housing options in areas with lower flood risk and chat with an AI expert on reducing risk. </p>
          </div>
          <div className={styles.feature}>
            <h3>Enterprise</h3>
            <p>Access advanced urban planning tools, analyze critical wetlands, zone buildings sustainably </p>
          </div>
          <div className={styles.feature}>
            <h3>API</h3>
            <p>Integrate risk analysis directly into your applications. For government, insurance, and building developers</p>
          </div>
        </div>

        <div className={styles.aboutSection}>
          <h2 className={styles.aboutTitle}>About the Project</h2>
          <p className={styles.aboutText}>
            The Multimodal Flood Risk Analysis Platform is an advanced solution designed to empower individuals,
            city planners, and enterprises with actionable insights into flood hazards and enhance sustainability. By leveraging custom
            machine learning and high-resolution spatial data, we aim to help build more resilient and sustainabile communities in the face
            of an evolving climate. {' '}
            <Link href="/about" style={{ color: '#000', fontWeight: 600, textDecoration: 'underline' }}>Read more</Link>
          </p>
        </div>
      </main>

      <style jsx global>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
