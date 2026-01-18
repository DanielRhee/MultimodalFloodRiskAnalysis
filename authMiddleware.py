from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
import os

security = HTTPBearer()

auth0Config = None

def loadAuth0Config():
    global auth0Config
    if auth0Config:
        return auth0Config
    
    credentials = {}
    credPath = os.path.join(os.path.dirname(__file__), "dbCredentials.txt")
    with open(credPath, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                credentials[key.strip()] = value.strip()
    
    auth0Config = {
        "domain": credentials.get("auth0_domain"),
        "audience": credentials.get("auth0_identifier"),
        "clientId": credentials.get("auth0_client_id")
    }
    return auth0Config

def getJwksClient():
    config = loadAuth0Config()
    jwksUrl = f"https://{config['domain']}/.well-known/jwks.json"
    return PyJWKClient(jwksUrl)

def verifyToken(token: str):
    config = loadAuth0Config()
    
    try:
        jwksClient = getJwksClient()
        signingKey = jwksClient.get_signing_key_from_jwt(token)
        
        payload = jwt.decode(
            token,
            signingKey.key,
            algorithms=["RS256"],
            audience=config["audience"],
            issuer=f"https://{config['domain']}/"
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )

async def getCurrentUser(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = verifyToken(token)
    
    return {
        "auth0Id": payload.get("sub"),
        "email": payload.get("email", payload.get("sub")),
        "name": payload.get("name", payload.get("nickname", "User"))
    }

def optionalAuth(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))):
    if credentials is None:
        return None
    try:
        token = credentials.credentials
        payload = verifyToken(token)
        return {
            "auth0Id": payload.get("sub"),
            "email": payload.get("email", payload.get("sub")),
            "name": payload.get("name", payload.get("nickname", "User"))
        }
    except:
        return None
