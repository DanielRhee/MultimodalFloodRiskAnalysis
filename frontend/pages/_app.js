import '@/styles/globals.css';
import { Inter } from 'next/font/google';
import { Auth0Provider } from '@auth0/auth0-react';
import { AuthProvider } from '@/context/authContext';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

const auth0Config = {
  domain: "dev-ifsoyomjc0reumg2.us.auth0.com",
  clientId: "INva2JsS3eZ9smYX9dmTwf1bgsSnlJbZ",
  authorizationParams: {
    redirect_uri: typeof window !== 'undefined' ? window.location.origin : '',
    audience: "https://api.myapp.com"
  }
};

export default function App({ Component, pageProps }) {
  return (
    <Auth0Provider {...auth0Config}>
      <AuthProvider>
        <div className={inter.variable}>
          <Component {...pageProps} />
        </div>
      </AuthProvider>
    </Auth0Provider>
  );
}

