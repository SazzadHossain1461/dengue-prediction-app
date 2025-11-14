// FILE 10: frontend/src/pages/Predict.js

import React, { useState } from 'react';
import { 
  Card, 
  CardContent, 
  TextField, 
  Button, 
  Grid, 
  Typography, 
  Box, 
  Alert, 
  CircularProgress, 
  Select, 
  MenuItem, 
  FormControl, 
  InputLabel,
  Stepper,
  Step,
  StepLabel
} from '@mui/material';
import { useForm, Controller } from 'react-hook-form';
import { api } from '../api/services';

const steps = ['Patient Information', 'Test Results', 'Environmental Factors', 'Prediction Result'];

function Predict() {
  const { control, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      gender: 'Male',
      age: 30,
      ns1: 0,
      igg: 0,
      igm: 0,
      area: 'Mirpur',
      area_type: 'Developed',
      house_type: 'Building',
      district: 'Dhaka',
    },
  });

  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);

  const watchedValues = watch();

  const onSubmit = async (data) => {
    setLoading(true);
    setError(null);
    
    const response = await api.createPrediction(data);
    
    if (response.error) {
      setError(response.message);
    } else {
      setResult(response.data);
      setActiveStep(3);
    }
    
    setLoading(false);
  };

  const handleNext = (e) => {
    e.preventDefault();
    if (isStepValid()) {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleReset = () => {
    setActiveStep(0);
    setResult(null);
    setError(null);
    reset();
  };

  const isStepValid = () => {
    switch (activeStep) {
      case 0:
        return watchedValues.age && watchedValues.age >= 1 && watchedValues.age <= 100;
      case 1:
        return watchedValues.ns1 !== undefined && watchedValues.igg !== undefined && watchedValues.igm !== undefined;
      case 2:
        return watchedValues.district && watchedValues.area && watchedValues.area_type && watchedValues.house_type;
      default:
        return true;
    }
  };

  const getStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="gender" 
                control={control} 
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Gender</InputLabel>
                    <Select {...field} label="Gender">
                      <MenuItem value="Male">Male</MenuItem>
                      <MenuItem value="Female">Female</MenuItem>
                    </Select>
                  </FormControl>
                )} 
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="age" 
                control={control} 
                render={({ field }) => (
                  <TextField 
                    {...field} 
                    label="Age" 
                    type="number" 
                    fullWidth 
                    inputProps={{ min: 1, max: 100 }}
                    onChange={(e) => field.onChange(parseInt(e.target.value) || 0)}
                  />
                )} 
              />
            </Grid>
          </Grid>
        );
      
      case 1:
        return (
          <Grid container spacing={2}>
            {['ns1', 'igg', 'igm'].map((test) => (
              <Grid item xs={12} sm={4} key={test}>
                <Controller 
                  name={test} 
                  control={control} 
                  render={({ field }) => (
                    <FormControl fullWidth>
                      <InputLabel>
                        {test === 'ns1' ? 'NS1 Antigen' : 
                         test === 'igg' ? 'IgG Antibody' : 'IgM Antibody'}
                      </InputLabel>
                      <Select 
                        {...field} 
                        label={test === 'ns1' ? 'NS1 Antigen' : 
                               test === 'igg' ? 'IgG Antibody' : 'IgM Antibody'}
                        onChange={(e) => field.onChange(parseInt(e.target.value))}
                      >
                        <MenuItem value={0}>Negative</MenuItem>
                        <MenuItem value={1}>Positive</MenuItem>
                      </Select>
                    </FormControl>
                  )} 
                />
              </Grid>
            ))}
          </Grid>
        );
      
      case 2:
        return (
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="district" 
                control={control} 
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>District</InputLabel>
                    <Select {...field} label="District">
                      <MenuItem value="Dhaka">Dhaka</MenuItem>
                      <MenuItem value="Mirpur">Mirpur</MenuItem>
                      <MenuItem value="Jatrabari">Jatrabari</MenuItem>
                      <MenuItem value="Demra">Demra</MenuItem>
                      <MenuItem value="Kamrangirchar">Kamrangirchar</MenuItem>
                      <MenuItem value="Hazaribagh">Hazaribagh</MenuItem>
                    </Select>
                  </FormControl>
                )} 
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="area" 
                control={control} 
                render={({ field }) => (
                  <TextField {...field} label="Area/Location" fullWidth />
                )} 
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="area_type" 
                control={control} 
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Area Type</InputLabel>
                    <Select {...field} label="Area Type">
                      <MenuItem value="Developed">Developed</MenuItem>
                      <MenuItem value="Undeveloped">Undeveloped</MenuItem>
                      <MenuItem value="Rural">Rural</MenuItem>
                    </Select>
                  </FormControl>
                )} 
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller 
                name="house_type" 
                control={control} 
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>House Type</InputLabel>
                    <Select {...field} label="House Type">
                      <MenuItem value="Building">Building</MenuItem>
                      <MenuItem value="Apartment">Apartment</MenuItem>
                      <MenuItem value="Tinshed">Tinshed</MenuItem>
                      <MenuItem value="Slum">Slum</MenuItem>
                      <MenuItem value="Other">Other</MenuItem>
                    </Select>
                  </FormControl>
                )} 
              />
            </Grid>
          </Grid>
        );
      
      case 3:
        return (
          <Box>
            {result && (
              <Card sx={{ 
                background: result.risk_level === 'High' 
                  ? 'linear-gradient(135deg, #f5576c 0%, #f093fb 100%)' 
                  : 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', 
                color: 'white',
                textAlign: 'center',
                py: 4
              }}>
                <CardContent>
                  <Typography variant="h2" sx={{ mb: 2, fontWeight: 'bold' }}>
                    {result.risk_level} RISK
                  </Typography>
                  <Typography variant="h5" sx={{ mb: 3 }}>
                    Confidence: {((result.confidence || 0) * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    {result.message || 'Prediction completed'}
                  </Typography>
                  <Typography variant="caption" sx={{ opacity: 0.8 }}>
                    Model: {result.model_type || 'Hybrid AI Model'}
                  </Typography>
                </CardContent>
              </Card>
            )}
          </Box>
        );
      
      default:
        return 'Unknown step';
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 3, fontWeight: 'bold' }}>
        Dengue Risk Assessment
      </Typography>

      <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
        {steps.map((label) => (
          <Step key={label}>
            <StepLabel>{label}</StepLabel>
          </Step>
        ))}
      </Stepper>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              {error && (
                <Alert severity="error" sx={{ mb: 2 }}>
                  {error}
                </Alert>
              )}
              
              <form onSubmit={handleSubmit(onSubmit)}>
                {getStepContent(activeStep)}
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                  <Button
                    disabled={activeStep === 0}
                    onClick={handleBack}
                  >
                    Back
                  </Button>
                  
                  <Box>
                    {activeStep === steps.length - 1 ? (
                      <Button onClick={handleReset} variant="outlined">
                        New Assessment
                      </Button>
                    ) : activeStep === steps.length - 2 ? (
                      <Button 
                        type="submit" 
                        variant="contained" 
                        disabled={loading || !isStepValid()}
                      >
                        {loading ? <CircularProgress size={24} /> : 'Get Prediction'}
                      </Button>
                    ) : (
                      <Button 
                        variant="contained" 
                        onClick={handleNext}
                        disabled={!isStepValid()}
                      >
                        Next
                      </Button>
                    )}
                  </Box>
                </Box>
              </form>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          {/* Preview Card */}
          {activeStep < 3 && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Input Preview
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Gender:</strong> {String(watchedValues.gender || '')}
                </Typography>
                <Typography variant="body2" sx={{ mb: 1 }}>
                  <strong>Age:</strong> {String(watchedValues.age || '')}
                </Typography>
                {activeStep >= 1 && (
                  <>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>NS1:</strong> {watchedValues.ns1 ? 'Positive' : 'Negative'}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>IgG:</strong> {watchedValues.igg ? 'Positive' : 'Negative'}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>IgM:</strong> {watchedValues.igm ? 'Positive' : 'Negative'}
                    </Typography>
                  </>
                )}
                {activeStep >= 2 && (
                  <>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>District:</strong> {String(watchedValues.district || '')}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Area:</strong> {String(watchedValues.area || '')}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Area Type:</strong> {String(watchedValues.area_type || '')}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>House Type:</strong> {String(watchedValues.house_type || '')}
                    </Typography>
                  </>
                )}
              </CardContent>
            </Card>
          )}
        </Grid>
      </Grid>
    </Box>
  );
}

export default Predict;