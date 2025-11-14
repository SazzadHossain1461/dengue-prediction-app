// FILE 8: frontend/src/App.js

import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { 
  ThemeProvider, 
  createTheme, 
  CssBaseline, 
  Box, 
  AppBar, 
  Toolbar, 
  Typography, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  IconButton 
} from '@mui/material';
import { 
  Menu, 
  Dashboard as DashboardIcon, 
  Add, 
  List as ListIcon, 
  People 
} from '@mui/icons-material';
import DashboardPage from './pages/DashboardPage';
import Predict from './pages/Predict';
import PredictionsList from './pages/PredictionsList';
import PatientsList from './pages/PatientsList';

const theme = createTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#dc3545' },
  },
});

function App() {
  const [drawerOpen, setDrawerOpen] = useState(false);

  const handleDrawerClose = () => {
    setDrawerOpen(false);
  };

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
    { text: 'New Prediction', icon: <Add />, path: '/predict' },
    { text: 'Predictions', icon: <ListIcon />, path: '/predictions' },
    { text: 'Patients', icon: <People />, path: '/patients' },
  ];

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <BrowserRouter>
        <Box sx={{ display: 'flex' }}>
          {/* Header */}
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
              <IconButton 
                color="inherit" 
                onClick={() => setDrawerOpen(!drawerOpen)}
                edge="start"
              >
                <Menu />
              </IconButton>
              <Typography variant="h6" sx={{ flexGrow: 1, ml: 2 }}>
                🦟 Dengue Prediction System
              </Typography>
            </Toolbar>
          </AppBar>

          {/* Sidebar */}
          <Drawer 
            variant="temporary"
            open={drawerOpen} 
            onClose={handleDrawerClose}
            sx={{
              '& .MuiDrawer-paper': { 
                width: 250,
                boxSizing: 'border-box',
              },
            }}
          >
            <Box sx={{ width: 250, pt: 8 }}>
              <List>
                {menuItems.map((item) => (
                  <ListItem 
                    button 
                    key={item.text}
                    component={Link}
                    to={item.path}
                    onClick={handleDrawerClose}
                    sx={{
                      '&:hover': {
                        backgroundColor: 'primary.light',
                        color: 'white',
                      },
                    }}
                  >
                    <ListItemIcon sx={{ color: 'inherit' }}>
                      {item.icon}
                    </ListItemIcon>
                    <ListItemText primary={item.text} />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Drawer>

          {/* Main Content */}
          <Box 
            component="main" 
            sx={{ 
              flexGrow: 1, 
              p: 3, 
              mt: 8,
              width: { sm: `calc(100% - 250px)` },
              ml: { sm: '250px' },
            }}
          >
            <Routes>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/predictions" element={<PredictionsList />} />
              <Route path="/patients" element={<PatientsList />} />
            </Routes>
          </Box>
        </Box>
      </BrowserRouter>
    </ThemeProvider>
  );
}

export default App;