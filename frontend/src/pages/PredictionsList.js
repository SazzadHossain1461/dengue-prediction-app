// FILE 11: frontend/src/pages/PredictionsList.js

import React, { useState, useEffect } from 'react';
import { 
  Table, 
  TableBody, 
  TableCell, 
  TableContainer, 
  TableHead, 
  TableRow, 
  Paper, 
  Typography, 
  CircularProgress, 
  Button, 
  Box, 
  Chip,
  Alert
} from '@mui/material';
import { Delete } from '@mui/icons-material';
import { api } from '../api/services';

function PredictionsList() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadPredictions();
  }, []);

  const loadPredictions = async () => {
    const response = await api.getPredictions();
    
    if (response.error) {
      setError(response.message);
    } else {
      setPredictions(Array.isArray(response.data) ? response.data : []);
    }
    
    setLoading(false);
  };

  const handleDelete = async (id) => {
    const response = await api.deletePrediction(id);
    
    if (!response.error) {
      setPredictions(predictions.filter(p => p.id !== id));
    } else {
      setError(response.message);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3 }}>Predictions</Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <TableContainer component={Paper}>
        <Table>
          <TableHead sx={{ backgroundColor: '#f5f5f5' }}>
            <TableRow>
              <TableCell>ID</TableCell>
              <TableCell>Age</TableCell>
              <TableCell>Gender</TableCell>
              <TableCell>NS1</TableCell>
              <TableCell>Risk Level</TableCell>
              <TableCell>Confidence</TableCell>
              <TableCell>Date</TableCell>
              <TableCell>Action</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {predictions.length === 0 ? (
              <TableRow>
                <TableCell colSpan={8} align="center">
                  <Typography variant="body1" sx={{ py: 2 }}>
                    No predictions found
                  </Typography>
                </TableCell>
              </TableRow>
            ) : (
              predictions.map((pred) => (
                <TableRow key={pred.id}>
                  <TableCell>{String(pred.id || '')}</TableCell>
                  <TableCell>{String(pred.age || '')}</TableCell>
                  <TableCell>{String(pred.gender || '')}</TableCell>
                  <TableCell>{pred.ns1 === 1 ? '✓' : '✗'}</TableCell>
                  <TableCell>
                    <Chip 
                      label={pred.prediction === 1 ? 'High' : 'Low'} 
                      color={pred.prediction === 1 ? 'error' : 'success'} 
                      size="small" 
                    />
                  </TableCell>
                  <TableCell>{((pred.confidence || 0) * 100).toFixed(1)}%</TableCell>
                  <TableCell>
                    {pred.created_at ? new Date(pred.created_at).toLocaleDateString() : 'N/A'}
                  </TableCell>
                  <TableCell>
                    <Button 
                      size="small" 
                      startIcon={<Delete />} 
                      onClick={() => handleDelete(pred.id)} 
                      color="error"
                    >
                      Delete
                    </Button>
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>
    </Box>
  );
}

export default PredictionsList;