from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from gridfs import GridFS
from bson import ObjectId
from datetime import datetime
import os

mongoClient = None
db = None
gridFs = None

def loadCredentials():
    credentials = {}
    credPath = os.path.join(os.path.dirname(__file__), "dbCredentials.txt")
    with open(credPath, "r") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, value = line.split("=", 1)
                credentials[key.strip()] = value.strip()
    return credentials

def initDatabase():
    global mongoClient, db, gridFs
    
    credentials = loadCredentials()
    mongoUri = credentials.get("mongodb_uri")
    
    if not mongoUri:
        raise ValueError("MongoDB URI not found in credentials")
    
    mongoClient = MongoClient(mongoUri)
    
    try:
        mongoClient.admin.command("ping")
        print("MongoDB connection successful")
    except ConnectionFailure as e:
        print(f"MongoDB connection failed: {e}")
        raise
    
    db = mongoClient["floodRiskAnalysis"]
    gridFs = GridFS(db)
    
    db.users.create_index("auth0Id", unique=True)
    db.projects.create_index("userId")
    db.projects.create_index([("userId", 1), ("createdAt", -1)])
    
    return db

def getDatabase():
    global db
    if db is None:
        initDatabase()
    return db

def getGridFs():
    global gridFs
    if gridFs is None:
        initDatabase()
    return gridFs

def createUser(auth0Id, email, name, userType="consumer"):
    usersCollection = getDatabase().users
    
    existingUser = usersCollection.find_one({"auth0Id": auth0Id})
    if existingUser:
        usersCollection.update_one(
            {"auth0Id": auth0Id},
            {"$set": {"email": email, "name": name, "lastLogin": datetime.utcnow()}}
        )
        return str(existingUser["_id"])
    
    result = usersCollection.insert_one({
        "auth0Id": auth0Id,
        "email": email,
        "name": name,
        "userType": userType,
        "createdAt": datetime.utcnow(),
        "lastLogin": datetime.utcnow()
    })
    return str(result.inserted_id)

def getUserByAuth0Id(auth0Id):
    usersCollection = getDatabase().users
    user = usersCollection.find_one({"auth0Id": auth0Id})
    if user:
        user["_id"] = str(user["_id"])
    return user

def createProject(userId, name, projectType):
    projectsCollection = getDatabase().projects
    
    result = projectsCollection.insert_one({
        "userId": userId,
        "name": name,
        "projectType": projectType,
        "satelliteFileId": None,
        "depthMapFileId": None,
        "analysisResult": None,
        "annotations": [],
        "chatHistory": [],
        "createdAt": datetime.utcnow(),
        "updatedAt": datetime.utcnow()
    })
    return str(result.inserted_id)

def getProjectsByUserId(userId):
    projectsCollection = getDatabase().projects
    projects = list(projectsCollection.find(
        {"userId": userId},
        {"name": 1, "projectType": 1, "createdAt": 1, "updatedAt": 1}
    ).sort("updatedAt", -1))
    
    for project in projects:
        project["_id"] = str(project["_id"])
    
    return projects

def getProjectById(projectId):
    projectsCollection = getDatabase().projects
    
    try:
        project = projectsCollection.find_one({"_id": ObjectId(projectId)})
        if project:
            project["_id"] = str(project["_id"])
        return project
    except:
        return None

def updateProject(projectId, updates):
    projectsCollection = getDatabase().projects
    updates["updatedAt"] = datetime.utcnow()
    
    projectsCollection.update_one(
        {"_id": ObjectId(projectId)},
        {"$set": updates}
    )

def deleteProject(projectId):
    project = getProjectById(projectId)
    if not project:
        return False
    
    fs = getGridFs()
    if project.get("satelliteFileId"):
        try:
            fs.delete(ObjectId(project["satelliteFileId"]))
        except:
            pass
    if project.get("depthMapFileId"):
        try:
            fs.delete(ObjectId(project["depthMapFileId"]))
        except:
            pass
    
    projectsCollection = getDatabase().projects
    projectsCollection.delete_one({"_id": ObjectId(projectId)})
    return True

def saveFileToGridFs(fileData, filename, contentType):
    fs = getGridFs()
    fileId = fs.put(fileData, filename=filename, contentType=contentType)
    return str(fileId)

def getFileFromGridFs(fileId):
    fs = getGridFs()
    try:
        gridFile = fs.get(ObjectId(fileId))
        return {
            "data": gridFile.read(),
            "filename": gridFile.filename,
            "contentType": gridFile.content_type
        }
    except:
        return None

def updateProjectAnnotations(projectId, annotations):
    updateProject(projectId, {"annotations": annotations})

def updateProjectChatHistory(projectId, chatHistory):
    updateProject(projectId, {"chatHistory": chatHistory})

def updateProjectAnalysis(projectId, satelliteFileId=None, depthMapFileId=None, analysisResult=None, riskMapFileId=None, landClassificationFileId=None, metrics=None, riskMaskFileId=None):
    updates = {}
    if satelliteFileId:
        updates["satelliteFileId"] = satelliteFileId
    if depthMapFileId:
        updates["depthMapFileId"] = depthMapFileId
    if analysisResult:
        updates["analysisResult"] = analysisResult
    if riskMapFileId:
        updates["riskMapFileId"] = riskMapFileId
    if landClassificationFileId:
        updates["landClassificationFileId"] = landClassificationFileId
    if metrics:
        updates["analysisMetrics"] = metrics
    if riskMaskFileId:
        updates["riskMaskFileId"] = riskMaskFileId
        
    if updates:
        updateProject(projectId, updates)

