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
        <div className={styles.brand}>Flood Risk Analysis</div>
        <nav style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
          <Link href="/help" style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>Help</Link>
          <a href="https://github.com/danielrhee/MultimodalFloodRiskAnalysis" target="_blank" rel="noopener" style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>GitHub</a>

          {isLoading ? (
            <Loader size={16} style={{ animation: 'spin 1s linear infinite' }} />
          ) : isAuthenticated ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <span style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>{user?.name || user?.email}</span>
              <button
                onClick={logout}
                style={{
                  display: 'flex', alignItems: 'center', gap: '0.25rem',
                  background: 'none', border: '1px solid var(--border)', borderRadius: '4px',
                  padding: '0.4rem 0.75rem', fontSize: '0.85rem', cursor: 'pointer',
                  color: 'var(--text-dim)'
                }}
              >
                <LogOut size={14} /> Logout
              </button>
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
              <LogIn size={14} /> Login
            </button>
          )}
        </nav>
      </header>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Multimodal Flood Risk<br />Analysis Platform
        </h1>

        <p className={styles.subtitle}>
          A simple, efficient tool combining satellite imagery and depth maps to predict flood risks in real-time.
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
            <p>Utilizes advanced computer vision to detect water bodies and elevation risks.</p>
          </div>
          <div className={styles.feature}>
            <h3>Instant Feedback</h3>
            <p>Get immediate risk assessments processed locally or via high-speed cloud APIs.</p>
          </div>
          <div className={styles.feature}>
            <h3>Secure Processing</h3>
            <p>Your data is processed securely and efficiently without unnecessary retention.</p>
          </div>
        </div>
      </main>

      <style jsx global>{`
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}

