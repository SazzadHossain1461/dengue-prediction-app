// FILE 7: frontend/src/api/services.js

import client from './client';

// Enhanced error handling
const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    return {
      error: true,
      message: error.response.data?.detail || error.response.data?.message || 'Server error occurred',
      status: error.response.status,
      data: null
    };
  } else if (error.request) {
    // Request made but no response received
    return {
      error: true,
      message: 'No response from server. Please check your connection.',
      status: null,
      data: null
    };
  } else {
    // Something else happened
    return {
      error: true,
      message: error.message || 'An unexpected error occurred',
      status: null,
      data: null
    };
  }
};

const apiCall = async (apiFunction, ...args) => {
  try {
    const response = await apiFunction(...args);
    return {
      error: false,
      data: response.data,
      status: response.status
    };
  } catch (error) {
    return handleApiError(error);
  }
};

export const api = {
  // Health & System
  health: () => apiCall(client.get, '/api/health'),
  
  // Predictions
  createPrediction: (data) => apiCall(client.post, '/api/predictions', data),
  getPredictions: (skip = 0, limit = 100) => apiCall(client.get, '/api/predictions', { params: { skip, limit } }),
  getPrediction: (id) => apiCall(client.get, `/api/predictions/${id}`),
  deletePrediction: (id) => apiCall(client.delete, `/api/predictions/${id}`),
  
  // Patients
  createPatient: (data) => apiCall(client.post, '/api/patients', data),
  getPatients: (skip = 0, limit = 100) => apiCall(client.get, '/api/patients', { params: { skip, limit } }),
  getPatient: (id) => apiCall(client.get, `/api/patients/${id}`),
  deletePatient: (id) => apiCall(client.delete, `/api/patients/${id}`),
  
  // Analytics
  getSummary: () => apiCall(client.get, '/api/analytics/summary'),
  getTrends: (days = 30) => apiCall(client.get, '/api/analytics/trends', { params: { days } }),
  
  // Model & Dataset Info
  getModelInfo: () => apiCall(client.get, '/api/model/info'),
  getDatasetInfo: () => apiCall(client.get, '/api/dataset/info'),
  retrainModel: () => apiCall(client.post, '/api/model/retrain'),
};