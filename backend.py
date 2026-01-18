from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Depends
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import uuid
from datetime import datetime, timedelta
from typing import Optional, List
import io
from google import genai
import os
import asyncio
from functools import partial

from floodRiskService import FloodRiskService
from database import initDatabase, createUser, getUserByAuth0Id, createProject, getProjectsByUserId, getProjectById, updateProject, deleteProject, saveFileToGridFs, getFileFromGridFs, updateProjectAnnotations, updateProjectChatHistory, updateProjectAnalysis, updateUserChatHistory
from authMiddleware import getCurrentUser, optionalAuth

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
geminiApiKey = None
geniaiClient = None
consumerSystemPrompt = None

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

    try:
        initDatabase()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization failed: {e}")

    global geminiApiKey, geniaiClient
    try:
        with open("geminiApiKey.txt", "r") as f:
            geminiApiKey = f.read().strip()
            geniaiClient = genai.Client(api_key=geminiApiKey)
            print("Gemini API Key loaded.")
    except Exception as e:
        print(f"Error loading Gemini API key: {e}")

    global consumerSystemPrompt
    try:
        with open("consumerSystemPrompt.txt", "r") as f:
            consumerSystemPrompt = f.read().strip()
            print("Consumer system prompt loaded.")
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        consumerSystemPrompt = "You are a helpful assistant."

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
    stride: int = Form(128),
    projectId: Optional[str] = Form(None),
    saveEnabled: bool = Form(False),
    user: dict = Depends(optionalAuth)
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

        loop = asyncio.get_event_loop()

        # Run heavy analysis in thread pool
        result = await loop.run_in_executor(
            None,
            partial(service.analyze, imagePIL, depthMap=depthMapProcessed, tileSize=tileSize, stride=stride)
        )

        # Run visualization generation in thread pool
        visualizationBytes = await loop.run_in_executor(
            None,
            partial(service.generateVisualization, imagePIL, result['riskMask'], result['landMask'])
        )

        riskMapBytes = await loop.run_in_executor(
            None,
            service.generateRiskMapBytes, result['riskMask']
        )

        landClassBytes = await loop.run_in_executor(
            None,
            service.generateLandClassificationBytes, result['landMask']
        )

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

        if saveEnabled and projectId and user:
            # Check ownership before saving
            dbUser = getUserByAuth0Id(user["auth0Id"])
            if dbUser:
                project = getProjectById(projectId)
                if project and project["userId"] == dbUser["_id"]:
                    riskMapFileId = saveFileToGridFs(riskMapBytes, f"risk_map_{analysisId}.png", "image/png")
                    landClassFileId = saveFileToGridFs(landClassBytes, f"land_class_{analysisId}.png", "image/png")

                    # Save raw risk mask for future calculations
                    maskBuffer = io.BytesIO()
                    np.save(maskBuffer, result['riskMask'])
                    maskBuffer.seek(0)
                    riskMaskFileId = saveFileToGridFs(maskBuffer.read(), f"risk_mask_{analysisId}.npy", "application/octet-stream")

                    metrics = {
                        'averageRisk': result['averageRisk'],
                        'riskByLandClass': result['riskByLandClass'],
                        'landClassDistribution': result['landClassDistribution']
                    }

                    updateProjectAnalysis(
                        projectId,
                        riskMapFileId=riskMapFileId,
                        landClassificationFileId=landClassFileId,
                        metrics=metrics,
                        riskMaskFileId=riskMaskFileId
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
def getRiskPoint(analysisId: str, pctX: float, pctY: float):
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    if result.riskMask is None:
        raise HTTPException(status_code=404, detail="Risk data not available")

    height, width = result.riskMask.shape

    # Calculate center pixel from percentages
    centerX = int(pctX * width)
    centerY = int(pctY * height)

    # Ensure center is within bounds
    centerX = max(0, min(centerX, width - 1))
    centerY = max(0, min(centerY, height - 1))

    # Define window size (e.g., 5x5)
    windowRadius = 2
    yStart = max(0, centerY - windowRadius)
    yEnd = min(height, centerY + windowRadius + 1)
    xStart = max(0, centerX - windowRadius)
    xEnd = min(width, centerX + windowRadius + 1)

    # Extract window and calculate mean
    window = result.riskMask[yStart:yEnd, xStart:xEnd]
    riskValue = float(np.mean(window))

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

class PolygonVertex(BaseModel):
    x: float
    y: float

class PolygonRiskRequest(BaseModel):
    vertices: list[PolygonVertex]

@app.post("/analyze/{analysisId}/polygon-risk")
def getPolygonRisk(analysisId: str, request: PolygonRiskRequest):
    """Calculate average risk within a polygon defined by percentage vertices."""
    cleanExpiredCache()

    if analysisId not in analysisCache:
        raise HTTPException(status_code=404, detail="Analysis not found or expired")

    result = analysisCache[analysisId]
    if result.riskMask is None:
        raise HTTPException(status_code=404, detail="Risk data not available")

    if len(request.vertices) < 3:
        raise HTTPException(status_code=400, detail="Polygon must have at least 3 vertices")

    height, width = result.riskMask.shape

    # Convert percentage vertices to pixel coordinates
    pixelVertices = []
    for v in request.vertices:
        px = int(v.x * width)
        py = int(v.y * height)
        px = max(0, min(px, width - 1))
        py = max(0, min(py, height - 1))
        pixelVertices.append((px, py))

    # Create polygon mask using PIL
    from PIL import ImageDraw
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pixelVertices, fill=255)
    maskArray = np.array(mask)

    # Apply mask and calculate mean risk
    polygonPixels = result.riskMask[maskArray == 255]

    if len(polygonPixels) == 0:
        return {"averageRisk": 0.0, "pixelsAnalyzed": 0}

    averageRisk = float(np.mean(polygonPixels))

    return {
        "averageRisk": averageRisk,
        "pixelsAnalyzed": int(len(polygonPixels))
    }

@app.get("/projects/{projectId}/risk-point")
async def getProjectRiskPoint(projectId: str, pctX: float, pctY: float, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project or project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("riskMaskFileId"):
         raise HTTPException(status_code=400, detail="Analysis data not available for this project")

    try:
        maskFile = getFileFromGridFs(project["riskMaskFileId"])
        if not maskFile:
            raise HTTPException(status_code=404, detail="Risk mask file missing")

        with io.BytesIO(maskFile["data"]) as f:
            riskMask = np.load(f)

        height, width = riskMask.shape
        centerX = int(pctX * width)
        centerY = int(pctY * height)
        centerX = max(0, min(centerX, width - 1))
        centerY = max(0, min(centerY, height - 1))

        # 5x5 window
        windowRadius = 2
        yStart = max(0, centerY - windowRadius)
        yEnd = min(height, centerY + windowRadius + 1)
        xStart = max(0, centerX - windowRadius)
        xEnd = min(width, centerX + windowRadius + 1)

        window = riskMask[yStart:yEnd, xStart:xEnd]
        riskValue = float(np.mean(window))

        return {"risk": riskValue}
    except Exception as e:
        print(f"Error calculating project risk point: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{projectId}/polygon-risk")
async def getProjectPolygonRisk(projectId: str, request: PolygonRiskRequest, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project or project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.get("riskMaskFileId"):
         raise HTTPException(status_code=400, detail="Analysis data not available for this project")

    try:
        maskFile = getFileFromGridFs(project["riskMaskFileId"])
        if not maskFile:
            raise HTTPException(status_code=404, detail="Risk mask file missing")

        with io.BytesIO(maskFile["data"]) as f:
            riskMask = np.load(f)

        height, width = riskMask.shape
        pixelVertices = []
        for v in request.vertices:
            px = int(v.x * width)
            py = int(v.y * height)
            px = max(0, min(px, width - 1))
            py = max(0, min(py, height - 1))
            pixelVertices.append((px, py))

        from PIL import ImageDraw
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(pixelVertices, fill=255)
        maskArray = np.array(mask)

        polygonPixels = riskMask[maskArray == 255]

        if len(polygonPixels) == 0:
            return {"averageRisk": 0.0, "pixelsAnalyzed": 0}

        averageRisk = float(np.mean(polygonPixels))

        return {
            "averageRisk": averageRisk,
            "pixelsAnalyzed": int(len(polygonPixels))
        }
    except Exception as e:
        print(f"Error calculating project polygon risk: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ChatRequest(BaseModel):
    message: str
    history: list = []
    projectId: Optional[str] = None
    saveEnabled: bool = False

@app.post("/chat")
async def chatWithGemini(request: ChatRequest, user: dict = Depends(optionalAuth)):
    try:
        if geniaiClient is None:
            raise HTTPException(status_code=503, detail="Gemini API not initialized")

        chatHistory = []
        if consumerSystemPrompt:
             chatHistory.append({"role": "user", "parts": [{"text": consumerSystemPrompt}]})
             chatHistory.append({"role": "model", "parts": [{"text": "Understood. I will act as a helpful flood risk analysis assistant for homeowners."}]})

        for msg in request.history:
            role = "user" if msg['role'] == 'user' else "model"
            chatHistory.append({"role": role, "parts": [{"text": msg['content']}]})

        chat = geniaiClient.chats.create(model="gemini-2.5-flash", history=chatHistory)
        response = chat.send_message(request.message)

        if request.saveEnabled and user:
            newHistory = request.history + [
                {"role": "user", "content": request.message},
                {"role": "model", "content": response.text}
            ]
            if request.projectId:
                updateProjectChatHistory(request.projectId, newHistory)
            else:
                dbUser = getUserByAuth0Id(user["auth0Id"])
                if dbUser:
                    updateUserChatHistory(dbUser["_id"], newHistory)

        return {"response": response.text}
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Flood Risk Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "POST /analyze": "Submit image for flood risk analysis",
            "GET /analyze/{analysisId}/visualization": "Get combined visualization",
            "GET /analyze/{analysisId}/risk-map": "Get risk heatmap",
            "GET /analyze/{analysisId}/land-classification": "Get land classification",
            "POST /chat": "Chat with Gemini about flood risk",
            "GET /health": "Check API health status",
            "GET /me": "Get current user info",
            "GET /projects": "List user projects",
            "POST /projects": "Create new project",
            "GET /projects/{id}": "Get project details",
            "PUT /projects/{id}": "Update project",
            "DELETE /projects/{id}": "Delete project"
        }
    }

class CreateProjectRequest(BaseModel):
    name: str
    projectType: str

class UpdateProjectRequest(BaseModel):
    name: Optional[str] = None
    annotations: Optional[List[dict]] = None
    chatHistory: Optional[List[dict]] = None
    polygonVertices: Optional[List[dict]] = None

@app.get("/me")
async def getMe(user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        userId = createUser(user["auth0Id"], user["email"], user["name"])
        dbUser = getUserByAuth0Id(user["auth0Id"])
    return dbUser

@app.get("/projects")
async def listProjects(user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        createUser(user["auth0Id"], user["email"], user["name"])
        dbUser = getUserByAuth0Id(user["auth0Id"])

    projects = getProjectsByUserId(dbUser["_id"])
    return {"projects": projects}

@app.post("/projects")
async def createNewProject(request: CreateProjectRequest, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        createUser(user["auth0Id"], user["email"], user["name"])
        dbUser = getUserByAuth0Id(user["auth0Id"])

    projectId = createProject(dbUser["_id"], request.name, request.projectType)
    return {"projectId": projectId}

@app.get("/projects/{projectId}")
async def getProject(projectId: str, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    return project

@app.put("/projects/{projectId}")
async def updateProjectEndpoint(projectId: str, request: UpdateProjectRequest, user: dict = Depends(getCurrentUser)):
    print(f"Updating project {projectId} with data: {request.dict(exclude_unset=True)}")
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.annotations is not None:
        updates["annotations"] = request.annotations
    if request.chatHistory is not None:
        updates["chatHistory"] = request.chatHistory
    if request.polygonVertices is not None:
        updates["polygonVertices"] = request.polygonVertices

    if updates:
        updateProject(projectId, updates)

    return {"success": True}

@app.delete("/projects/{projectId}")
async def deleteProjectEndpoint(projectId: str, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    deleteProject(projectId)
    return {"success": True}

@app.post("/projects/{projectId}/files")
async def uploadProjectFile(
    projectId: str,
    fileType: str = Form(...),
    file: UploadFile = File(...),
    user: dict = Depends(getCurrentUser)
):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    fileData = await file.read()
    fileId = saveFileToGridFs(fileData, file.filename, file.content_type)

    if fileType == "satellite":
        updateProjectAnalysis(projectId, satelliteFileId=fileId)
    elif fileType == "depthMap":
        updateProjectAnalysis(projectId, depthMapFileId=fileId)

    return {"fileId": fileId}

@app.get("/projects/{projectId}/files/{fileId}")
async def getProjectFile(projectId: str, fileId: str, user: dict = Depends(getCurrentUser)):
    dbUser = getUserByAuth0Id(user["auth0Id"])
    if not dbUser:
        raise HTTPException(status_code=404, detail="User not found")

    project = getProjectById(projectId)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project["userId"] != dbUser["_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    fileData = getFileFromGridFs(fileId)
    if not fileData:
        raise HTTPException(status_code=404, detail="File not found")

    return Response(
        content=fileData["data"],
        media_type=fileData["contentType"],
        headers={"Content-Disposition": f"attachment; filename={fileData['filename']}"}
    )
