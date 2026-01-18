import Head from 'next/head';
import Link from 'next/link';
import { Layers, Activity, Database } from 'lucide-react';
import styles from '@/styles/Home.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Flood Risk Analytics</title>
        <meta name="description" content="Minimal Flood Risk Analysis Tool" />
      </Head>

      <header className={styles.header}>
        <div className={styles.brand}>Flood Risk Analysis</div>
        <nav style={{ display: 'flex', gap: '1.5rem' }}>
          <Link href="/help" style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>Help</Link>
          <a href="https://github.com/danielrhee/MultimodalFloodRiskAnalysis" target="_blank" rel="noopener" style={{ color: 'var(--text-dim)', fontSize: '0.9rem' }}>GitHub</a>
        </nav>
      </header>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Multimodal Flood Risk<br />Analysis Platform
        </h1>

        <p className={styles.subtitle}>
          A simple, efficient tool combining satellite imagery and depth maps to predict flood risks in real-time.
        </p>

        <Link href="/portal" className={styles.ctaButton}>
          Launch Portal
        </Link>

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
    </div>
  );
}
