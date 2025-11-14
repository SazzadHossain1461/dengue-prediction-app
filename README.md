# 🦟 Dengue Prediction System

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A comprehensive machine learning system for predicting dengue fever risk using advanced neural networks and real-time analytics.**

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## 📋 About

The **Dengue Prediction System** is a full-stack web application that leverages machine learning to predict dengue fever risk based on patient data and environmental factors. Built with modern technologies, it provides healthcare professionals with real-time predictions, comprehensive analytics, and intuitive patient management tools.

### Why This Project?

Dengue fever affects over 390 million people annually worldwide. Early risk detection is crucial for:
- ✅ Quick intervention and treatment
- ✅ Resource allocation
- ✅ Public health monitoring
- ✅ Epidemiological research

---

## 🌟 Features

### 📊 Analytics Dashboard
- Real-time statistics and metrics
- Interactive trend charts (30-day view)
- Risk distribution visualization
- Confidence score analysis

### 🔮 Prediction Engine
- ML-powered dengue risk prediction
- Multiple input parameters (age, antibodies, location, environment)
- Confidence scoring (0-100%)
- Risk level classification (High/Low)

### 👥 Patient Management
- Create and manage patient records
- Medical history tracking
- Demographics information
- Prediction history per patient

### 📈 Advanced Analytics
- Predictions by district
- Daily trend analysis
- Positive case tracking
- Average confidence metrics

### 🔌 REST API
- Full-featured REST API with 10+ endpoints
- Auto-generated Swagger documentation
- Health check endpoints
- CRUD operations for all entities

### 🎨 User Interface
- Modern Material-UI design
- Responsive layout (mobile, tablet, desktop)
- Dark/Light mode support
- Intuitive navigation

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.9 or higher
- **Node.js** 18 or higher
- **npm** or **yarn** package manager

### 1-Minute Setup

```bash
# Clone repository
git clone https://github.com/yourusername/dengue-prediction-system.git
cd dengue-prediction-system

# Setup backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup frontend
cd frontend
npm install

# Start backend (Terminal 1)
cd ..
python main.py

# Start frontend (Terminal 2)
cd frontend
npm start
```

**Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/dengue-prediction-system.git
cd dengue-prediction-system
```

### Step 2: Backend Setup

#### Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Update .env with your settings
DATABASE_URL=sqlite:///./dengue.db
DEBUG=True
API_TITLE=Dengue Prediction API
API_VERSION=1.0.0
MODEL_PATH=./models/model.h5
HOST=0.0.0.0
PORT=8000
```

#### Initialize Database

```bash
python main.py
# Press Ctrl+C after database initializes
```

### Step 3: Frontend Setup

```bash
cd frontend
npm install
```

### Step 4: Run Services

**Terminal 1 - Backend:**
```bash
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm start
```

---

## 📂 Project Structure

```
dengue-prediction-system/
│
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── database.py             # Database setup & ORM
│   ├── models.py               # SQLAlchemy models
│   ├── schemas.py              # Pydantic validation schemas
│   ├── crud.py                 # Database CRUD operations
│   ├── ml_utils.py             # ML model management
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Environment variables
│
├── 📁 frontend/
│   ├── 📁 public/
│   │   └── index.html          # Main HTML file
│   ├── 📁 src/
│   │   ├── 📁 api/
│   │   │   ├── client.js       # Axios HTTP client
│   │   │   └── services.js     # API service functions
│   │   ├── 📁 pages/
│   │   │   ├── Dashboard.js    # Analytics dashboard
│   │   │   ├── Predict.js      # Prediction form
│   │   │   ├── PredictionsList.js # View predictions
│   │   │   └── PatientsList.js # Manage patients
│   │   ├── App.js              # Main React component
│   │   ├── index.js            # React entry point
│   │   └── index.css           # Global styles
│   ├── package.json            # Node dependencies
│   └── .env                    # Frontend config
│
├── 📁 models/
│   └── model.h5                # TensorFlow model (auto-created)
│
├── 📄 README.md               # This file
├── 📄 LICENSE                 # MIT License
└── 📄 .gitignore              # Git ignore rules
```

## Dashboard
<img width="1918" height="922" alt="Screenshot (568)" src="https://github.com/user-attachments/assets/25744fb3-b204-49ac-8840-2d5fc48cbe79" />


## Prediction Assessment
<img width="1920" height="928" alt="Screenshot (569)" src="https://github.com/user-attachments/assets/3eacfb18-bf57-4f07-8d81-788d6570103f" />


## Results
<img width="1920" height="921" alt="Screenshot (570)" src="https://github.com/user-attachments/assets/0f7d2403-3cfd-49f6-a14f-1d396bc51f02" />

---

## 🔌 API Documentation

### Base URL
```
http://localhost:8000/api
```

### Authentication
Currently, the API uses open endpoints. Future versions will include JWT authentication.

### Endpoints

#### Health Check
```http
GET /api/health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "model": "loaded",
  "version": "1.0.0"
}
```

#### Predictions

**Create Prediction**
```http
POST /api/predictions
Content-Type: application/json

{
  "gender": "Male",
  "age": 35,
  "ns1": 1,
  "igg": 1,
  "igm": 0,
  "area": "Mirpur",
  "area_type": "Developed",
  "house_type": "Building",
  "district": "Dhaka",
  "patient_id": null
}
```

**Response:**
```json
{
  "id": 1,
  "prediction": 1,
  "confidence": 0.87,
  "risk_level": "High",
  "created_at": "2024-01-15T10:30:00"
}
```

**Get All Predictions**
```http
GET /api/predictions?skip=0&limit=100
```

**Get Specific Prediction**
```http
GET /api/predictions/{prediction_id}
```

**Delete Prediction**
```http
DELETE /api/predictions/{prediction_id}
```

#### Patients

**Create Patient**
```http
POST /api/patients
Content-Type: application/json

{
  "name": "John Doe",
  "gender": "Male",
  "age": 35,
  "area": "Mirpur",
  "district": "Dhaka",
  "phone": "+880123456789"
}
```

**Get All Patients**
```http
GET /api/patients?skip=0&limit=100
```

**Get Patient Details**
```http
GET /api/patients/{patient_id}
```

**Delete Patient**
```http
DELETE /api/patients/{patient_id}
```

#### Analytics

**Get Summary Statistics**
```http
GET /api/analytics/summary
```

**Response:**
```json
{
  "total_predictions": 150,
  "positive_cases": 45,
  "negative_cases": 105,
  "positive_percentage": 30.0,
  "average_confidence": 0.85
}
```

**Get Trends**
```http
GET /api/analytics/trends?days=30
```

### Interactive API Documentation

Access Swagger UI at: `http://localhost:8000/docs`

Access ReDoc at: `http://localhost:8000/redoc`

---

## 🗄️ Database Schema

### Patients Table
```sql
CREATE TABLE patients (
  id INTEGER PRIMARY KEY,
  name VARCHAR(100),
  gender VARCHAR(10),
  age INTEGER,
  area VARCHAR(100),
  district VARCHAR(100),
  phone VARCHAR(20),
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Predictions Table
```sql
CREATE TABLE predictions (
  id INTEGER PRIMARY KEY,
  patient_id INTEGER FOREIGN KEY,
  gender VARCHAR(10),
  age INTEGER,
  ns1 INTEGER,
  igg INTEGER,
  igm INTEGER,
  area VARCHAR(100),
  area_type VARCHAR(20),
  house_type VARCHAR(20),
  district VARCHAR(100),
  prediction INTEGER,
  confidence FLOAT,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🤖 Machine Learning Model

### Input Features
- **Age**: Patient age in years (0-120)
- **NS1**: NS1 antigen test result (0=Negative, 1=Positive)
- **IgG**: IgG antibody test result (0=Negative, 1=Positive)
- **IgM**: IgM antibody test result (0=Negative, 1=Positive)

### Model Architecture
```
Input Layer (4 features)
    ↓
Dense Layer (32 units, ReLU activation)
    ↓
Dense Layer (16 units, ReLU activation)
    ↓
Output Layer (1 unit, Sigmoid activation)
```

### Output
- **Prediction**: 0 (Negative) or 1 (Positive)
- **Confidence**: Probability score (0.0-1.0)
- **Risk Level**: "Low" or "High"

### Training Data
Built with dengue prediction dataset containing 1000+ patient records from Bangladesh regions.

---

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=sqlite:///./dengue.db

# Debug Mode
DEBUG=True

# API Configuration
API_TITLE=Dengue Prediction API
API_VERSION=1.0.0

# Model Configuration
MODEL_PATH=./models/model.h5

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### Database Options

#### SQLite (Default)
No additional setup required. Perfect for development.

```env
DATABASE_URL=sqlite:///./dengue.db
```

#### PostgreSQL
For production deployment:

```env
DATABASE_URL=postgresql://username:password@localhost:5432/dengue_db
```

Install PostgreSQL driver:
```bash
pip install psycopg2-binary
```

---

## 📊 Usage Examples

### Example 1: Create a Prediction

```bash
curl -X POST http://localhost:8000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 35,
    "ns1": 1,
    "igg": 1,
    "igm": 0,
    "area": "Mirpur",
    "area_type": "Developed",
    "house_type": "Building",
    "district": "Dhaka"
  }'
```

### Example 2: Get All Predictions

```bash
curl http://localhost:8000/api/predictions
```

### Example 3: Get Analytics Summary

```bash
curl http://localhost:8000/api/analytics/summary
```

### Example 4: Using Python Requests

```python
import requests

url = "http://localhost:8000/api/predictions"
data = {
    "gender": "Female",
    "age": 28,
    "ns1": 1,
    "igg": 0,
    "igm": 1,
    "area": "Dhanmondi",
    "area_type": "Developed",
    "house_type": "Building",
    "district": "Dhaka"
}

response = requests.post(url, json=data)
print(response.json())
```

---

## 🧪 Testing

### Run Backend Tests

```bash
pytest tests/
```

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/api/health

# All predictions
curl http://localhost:8000/api/predictions

# All patients
curl http://localhost:8000/api/patients

# Analytics summary
curl http://localhost:8000/api/analytics/summary
```

---

## 🐛 Troubleshooting

### Issue: Port Already in Use

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### Issue: Module Not Found

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: TensorFlow Error

**Solution:**
```bash
# Reinstall TensorFlow
pip install --upgrade tensorflow
```

### Issue: Frontend Won't Start

**Solution:**
```bash
cd frontend
npm cache clean --force
rm -rf node_modules
npm install
npm start
```

### Issue: Database Connection Error

**Solution:**
```bash
# Delete old database
rm dengue.db

# Restart backend
python main.py
```

---

## 🚀 Deployment

### Docker Deployment

```bash
docker-compose up -d
```

### Cloud Platforms

#### Heroku
```bash
git push heroku main
```

#### AWS EC2
```bash
ssh -i key.pem ubuntu@instance-ip
cd dengue-prediction-system
python main.py
```

#### DigitalOcean App Platform
1. Connect GitHub repository
2. Create deployment configuration
3. Deploy automatically

---

## 📈 Performance

### Benchmarks

| Metric | Value |
|--------|-------|
| API Response Time | < 500ms |
| Frontend Load Time | < 3s |
| Prediction Accuracy | 87% |
| Model AUC Score | 0.91 |
| Database Queries/sec | 1000+ |

## Confusion_matrix
<img width="2400" height="1800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/6453eb5b-bc15-4b06-a205-96c1ed8ce81b" />

## Training_history
<img width="3600" height="1200" alt="training_history" src="https://github.com/user-attachments/assets/6402feb9-a6d6-4117-bf9d-3ab97a6b80eb" />

---

## 🔒 Security Considerations

- ✅ Input validation with Pydantic
- ✅ SQL injection prevention (SQLAlchemy ORM)
- ✅ CORS headers configured
- ✅ Rate limiting ready (can be added)
- ✅ Environment variables for secrets
- ✅ HTTPS ready for production

### Production Checklist

- [ ] Set `DEBUG=False` in `.env`
- [ ] Use strong database passwords
- [ ] Enable HTTPS
- [ ] Configure CORS properly
- [ ] Set up monitoring and logging
- [ ] Regular database backups
- [ ] Update dependencies regularly

---

## 📚 Tech Stack

### Backend
- **Framework**: FastAPI 0.104
- **Server**: Uvicorn
- **Database**: SQLite/PostgreSQL
- **ORM**: SQLAlchemy
- **ML**: TensorFlow 2.14
- **Validation**: Pydantic
- **Language**: Python 3.9+

### Frontend
- **Framework**: React 18
- **UI Library**: Material-UI 5
- **HTTP Client**: Axios
- **Routing**: React Router 6
- **Charts**: Recharts
- **Forms**: React Hook Form
- **Language**: JavaScript/JSX

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

### 1. Fork the Repository
```bash
git clone https://github.com/yourusername/dengue-prediction-system.git
cd dengue-prediction-system
git remote add upstream https://github.com/original/dengue-prediction-system.git
```

### 2. Create Feature Branch
```bash
git checkout -b feature/amazing-feature
```

### 3. Make Changes and Commit
```bash
git add .
git commit -m "Add amazing feature"
```

### 4. Push to Branch
```bash
git push origin feature/amazing-feature
```

### 5. Open Pull Request
Create a Pull Request with a clear description of changes.

### Contribution Guidelines

- Follow PEP 8 style guide for Python
- Use meaningful variable names
- Add docstrings to functions
- Write tests for new features
- Update documentation
- Ensure code passes linting

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### You are free to:
- ✅ Use commercially
- ✅ Modify the software
- ✅ Distribute copies
- ✅ Use privately

### Under these conditions:
- 📋 Include license and copyright notice
- 📋 Disclose changes
- 📋 State significant changes

---

## 👥 Authors

**Developed by**: Sazzad Hossain

- GitHub: https://github.com/SazzadHossain1461
- Email: sazzadhossain74274@gmail.com
- LinkedIn: https://www.linkedin.com/in/sazzadhossain1461/

---

## 🙏 Acknowledgments

- TensorFlow & Keras teams for excellent ML framework
- FastAPI for modern Python web framework
- React team for powerful UI library
- Material-UI for beautiful components
- All contributors and supporters


---

## 📊 Project Statistics

![GitHub Stars](https://img.shields.io/github/stars/sazzadhossain/dengue-prediction-system?style=social)
![GitHub Forks](https://img.shields.io/github/forks/sazzadhossain/dengue-prediction-system?style=social)
![GitHub Issues](https://img.shields.io/github/issues/sazzadhossain/dengue-prediction-system)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/sazzadhossain/dengue-prediction-system)

---

## 🗺️ Roadmap

### Version 1.0 ✅
- [x] Core prediction engine
- [x] REST API
- [x] React frontend
- [x] Patient management
- [x] Analytics dashboard

### Version 1.1 (Planned)
- [ ] User authentication
- [ ] Email notifications
- [ ] Advanced analytics
- [ ] Mobile app
- [ ] Multi-language support

### Version 2.0 (Future)
- [ ] Model explainability (SHAP/LIME)
- [ ] Real-time predictions batch processing
- [ ] Integration with healthcare systems
- [ ] Advanced reporting

---

<div align="center">

### ⭐ If you find this project helpful, please give it a star!

**Made with ❤️ for better healthcare**

[Back to Top](#-dengue-prediction-system)

</div>
