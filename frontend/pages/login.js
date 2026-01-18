import Head from "next/head";
import { useAuth } from "@/context/authContext";
import { LogIn, Loader } from "lucide-react";
import styles from "@/styles/Home.module.css";

export default function Login() {
    const { loginWithRedirect, isLoading, isAuthenticated } = useAuth();

    if (isLoading) {
        return (
            <div className={styles.container}>
                <Head>
                    <title>Login | Flood Risk Analysis</title>
                </Head>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100vh" }}>
                    <Loader size={32} style={{ animation: "spin 1s linear infinite" }} />
                </div>
                <style jsx global>{`
                    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
                `}</style>
            </div>
        );
    }

    if (isAuthenticated) {
        if (typeof window !== "undefined") {
            window.location.href = "/";
        }
        return null;
    }

    return (
        <div className={styles.container}>
            <Head>
                <title>Login | Flood Risk Analysis</title>
                <meta name="description" content="Login to access Flood Risk Analysis Portal" />
            </Head>

            <header className={styles.header}>
                <div className={styles.brand}>Flood Risk Analysis</div>
            </header>

            <main className={styles.main}>
                <h1 className={styles.title}>
                    Welcome Back
                </h1>

                <p className={styles.subtitle}>
                    Sign in to access your flood risk analysis projects and saved data.
                </p>

                <button
                    className={styles.ctaButton}
                    onClick={() => loginWithRedirect()}
                    style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}
                >
                    <LogIn size={18} />
                    Sign In with Auth0
                </button>

                <div className={styles.features} style={{ marginTop: "3rem" }}>
                    <div className={styles.feature}>
                        <h3>Save Your Projects</h3>
                        <p>Your analysis results, annotations, and chat history are securely saved.</p>
                    </div>
                    <div className={styles.feature}>
                        <h3>Access Anywhere</h3>
                        <p>Log in from any device to continue where you left off.</p>
                    </div>
                    <div className={styles.feature}>
                        <h3>Enterprise Ready</h3>
                        <p>Advanced tools and collaboration features for planning teams.</p>
                    </div>
                </div>
            </main>
        </div>
    );
}
