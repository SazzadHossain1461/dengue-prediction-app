// FILE 9: frontend/src/pages/DashboardPage.js

import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Card, 
  CardContent, 
  Typography, 
  Box, 
  CircularProgress,
  Alert,
  Chip
} from '@mui/material';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { api } from '../api/services';

function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [trends, setTrends] = useState([]);
  const [modelInfo, setModelInfo] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    try {
      const [summaryRes, trendsRes, modelRes, datasetRes] = await Promise.all([
        api.getSummary(),
        api.getTrends(),
        api.getModelInfo(),
        api.getDatasetInfo()
      ]);
      
      // Handle API responses safely
      if (!summaryRes.error) setSummary(summaryRes.data);
      if (!trendsRes.error) setTrends(trendsRes.data || []);
      if (!modelRes.error) setModelInfo(modelRes.data);
      if (!datasetRes.error) setDatasetInfo(datasetRes.data);

      // Check for any errors
      const errors = [summaryRes, trendsRes, modelRes, datasetRes].filter(res => res.error);
      if (errors.length > 0) {
        setError(errors[0].message || 'Failed to load some data');
      }
    } catch (error) {
      console.error('Error loading data:', error);
      setError('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  const StatCard = ({ title, value, subtitle, color }) => (
    <Card sx={{ 
      background: color, 
      color: 'white',
      height: '100%',
      transition: 'transform 0.2s',
      '&:hover': { transform: 'translateY(-4px)' }
    }}>
      <CardContent>
        <Typography variant="h6" gutterBottom>{title}</Typography>
        <Typography variant="h3" sx={{ fontWeight: 'bold', mb: 1 }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" sx={{ opacity: 0.9 }}>
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  // Safe data extraction
  const totalPredictions = summary?.total_predictions || 0;
  const positiveCases = summary?.positive_cases || 0;
  const negativeCases = summary?.negative_cases || 0;
  const averageConfidence = summary?.average_confidence || 0;
  const positiveRate = summary?.positive_rate || '0';
  const negativeRate = summary?.negative_rate || '0';

  const caseDistribution = [
    { name: 'Positive', value: positiveCases },
    { name: 'Negative', value: negativeCases }
  ];

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 'bold' }}>
        Dashboard Overview
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {/* Model Info */}
      {modelInfo && (
        <Alert 
          severity={modelInfo.status === 'loaded' ? 'success' : 'warning'} 
          sx={{ mb: 3 }}
        >
          <strong>Model Status:</strong> {modelInfo.status} | 
          <strong> Type:</strong> {modelInfo.model_type}
        </Alert>
      )}

      {/* Dataset Info */}
      {datasetInfo && (
        <Alert severity="info" sx={{ mb: 3 }}>
          <strong>Dataset:</strong> {datasetInfo.dataset_size || 0} records | 
          <strong> Positive Rate:</strong> {datasetInfo.positive_rate || '0%'}
        </Alert>
      )}

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard 
            title="Total Predictions" 
            value={totalPredictions}
            color="linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard 
            title="Positive Cases" 
            value={positiveCases}
            subtitle={`${positiveRate}% of total`}
            color="linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard 
            title="Negative Cases" 
            value={negativeCases}
            subtitle={`${negativeRate}% of total`}
            color="linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard 
            title="Avg Confidence" 
            value={`${(averageConfidence * 100).toFixed(1)}%`}
            color="linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)"
          />
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3}>
        {/* Trends Chart */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Prediction Trends (Last 30 Days)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={trends}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="total" 
                    stroke="#8884d8" 
                    name="Total Predictions" 
                    strokeWidth={2}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="positive" 
                    stroke="#82ca9d" 
                    name="Positive Cases" 
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Distribution Chart */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2 }}>
                Case Distribution
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={caseDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    <Cell fill="#f5576c" />
                    <Cell fill="#4facfe" />
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Model Features */}
      {modelInfo?.features && Array.isArray(modelInfo.features) && (
        <Card sx={{ mt: 3 }}>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              Model Features
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {modelInfo.features.map((feature, index) => (
                <Chip 
                  key={index}
                  label={String(feature)}
                  variant="outlined"
                  color="primary"
                  size="small"
                />
              ))}
            </Box>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default DashboardPage;