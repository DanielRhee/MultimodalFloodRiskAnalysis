from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Optional
import io

from floodRiskService import FloodRiskService

app = FastAPI(title="Flood Risk Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = None
analysisCache = {}
CACHE_TTL_MINUTES = 60

class AnalysisResult:
    def __init__(self, data, timestamp, riskMask=None):
        self.data = data
        self.timestamp = timestamp
        self.riskMask = riskMask
        self.expiresAt = timestamp + timedelta(minutes=CACHE_TTL_MINUTES)

def cleanExpiredCache():
    now = datetime.now()
    expiredKeys = [key for key, result in analysisCache.items() if result.expiresAt < now]
    for key in expiredKeys:
        del analysisCache[key]

@app.on_event("startup")
async def startupEvent():
    global service
    print("Loading flood risk models...")
    service = FloodRiskService()
    print("Models loaded successfully")

@app.get("/health")
def healthCheck():
    return {
        "status": "healthy" if service is not None else "loading",
        "modelsLoaded": service is not None
    }

@app.post("/analyze")
async def analyzeFloodRisk(
    image: UploadFile = File(...),
    depthMap: Optional[UploadFile] = File(None),
    depthMin: Optional[float] = Form(None),
    depthMax: Optional[float] = Form(None),
    tileSize: int = Form(128),
    stride: int = Form(128)
):
    if service is None:
        raise HTTPException(status_code=503, detail="Models are still loading")

    cleanExpiredCache()

    try:
        imageBytes = await image.read()
        imagePIL = Image.open(io.BytesIO(imageBytes)).convert('RGB')

        depthMapProcessed = None
        if depthMap is not None:
            if depthMin is None or depthMax is None:
                raise HTTPException(
                    status_code=400,
                    detail="depthMin and depthMax are required when depthMap is provided"
                )

            depthBytes = await depthMap.read()
            depthPIL = Image.open(io.BytesIO(depthBytes)).convert('L')

            if depthPIL.size != imagePIL.size:
                depthPIL = depthPIL.resize(imagePIL.size, Image.Resampling.BILINEAR)

            depthArray = np.array(depthPIL).astype(np.float32)
            depthMapProcessed = depthMin + (depthArray / 255.0) * (depthMax - depthMin)
            depthMapProcessed = Image.fromarray(depthMapProcessed.astype(np.float32))

        result = service.analyze(imagePIL, depthMap=depthMapProcessed, tileSize=tileSize, stride=stride)

        visualizationBytes = service.generateVisualization(
            imagePIL,
            result['riskMask'],
            result['landMask']
        )
        riskMapBytes = service.generateRiskMapBytes(result['riskMask'])
        landClassBytes = service.generateLandClassificationBytes(result['landMask'])

        analysisId = str(uuid.uuid4())
        analysisCache[analysisId] = AnalysisResult(
            {
                'visualization': visualizationBytes,
                'riskMap': riskMapBytes,
                'landClassification': landClassBytes,
                'metadata': {
                    'averageRisk': result['averageRisk'],
                    'riskByLandClass': result['riskByLandClass'],
                    'landClassDistribution': result['landClassDistribution'],
                    'imageWidth': result['imageWidth'],
                    'imageHeight': result['imageHeight']
                }
            },
            datetime.now(),
            riskMask=result['riskMask']
        )

        return {
            'analysisId': analysisId,
            'averageRisk': result['averageRisk'],
            'riskByLandClass': result['riskByLandClass'],
            'landClassDistribution': result['landClassDistribution'],
            'imageWidth': result['imageWidth'],
            'imageHeight': result['imageHeight']
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/analyze/{analysisId}/risk-point")
def getRiskPoint(analysisId: str, x: int, y: int):
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    if result.riskMask is None:
        raise HTTPException(status_code=404, detail="Risk data not available")

    height, width = result.riskMask.shape
    
    if not (0 <= x < width and 0 <= y < height):
        raise HTTPException(status_code=400, detail="Coordinates out of bounds")

    riskValue = float(result.riskMask[y, x])
    return {"risk": riskValue}

@app.get("/analyze/{analysisId}/visualization")
def getVisualization(analysisId: str):
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    return Response(content=result.data['visualization'], media_type="image/png")

@app.get("/analyze/{analysisId}/risk-map")
def getRiskMap(analysisId: str):
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    return Response(content=result.data['riskMap'], media_type="image/png")

@app.get("/analyze/{analysisId}/land-classification")
def getLandClassification(analysisId: str):
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    return Response(content=result.data['landClassification'], media_type="image/png")

@app.get("/")
def root():
    return {
        "message": "Flood Risk Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "POST /analyze": "Submit image for flood risk analysis",
            "GET /analyze/{analysisId}/visualization": "Get combined visualization",
            "GET /analyze/{analysisId}/risk-map": "Get risk heatmap",
            "GET /analyze/{analysisId}/land-classification": "Get land classification",
            "GET /health": "Check API health status"
        }
    }
