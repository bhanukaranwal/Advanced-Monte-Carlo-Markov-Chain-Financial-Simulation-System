/**
 * MCMF Mobile Dashboard - React Native Application
 * Real-time portfolio monitoring, risk analytics, and interactive visualizations
 */

import React, { useState, useEffect, useRef } from 'react';
import {
  StyleSheet,
  Text,
  View,
  ScrollView,
  TouchableOpacity,
  Dimensions,
  Alert,
  RefreshControl,
  Platform,
  StatusBar,
  SafeAreaView,
  Animated,
  PanResponder
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import Icon from 'react-native-vector-icons/MaterialCommunityIcons';
import { LineChart, AreaChart, PieChart, BarChart } from 'react-native-chart-kit';
import PushNotification from 'react-native-push-notification';
import AsyncStorage from '@react-native-async-storage/async-storage';
import NetInfo from '@react-native-community/netinfo';
import { WebSocket } from 'react-native-websocket';
import LinearGradient from 'react-native-linear-gradient';

const { width, height } = Dimensions.get('window');
const Tab = createBottomTabNavigator();
const Stack = createStackNavigator();

// Theme configuration
const theme = {
  colors: {
    primary: '#667eea',
    secondary: '#764ba2',
    success: '#28a745',
    danger: '#dc3545',
    warning: '#ffc107',
    info: '#17a2b8',
    dark: '#343a40',
    light: '#f8f9fa',
    background: '#ffffff',
    surface: '#f5f5f5',
    text: '#212529',
    textSecondary: '#6c757d'
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32
  },
  borderRadius: 12,
  elevation: 4
};

// API Configuration
const API_CONFIG = {
  baseURL: 'https://api.mcmf-system.com',
  websocketURL: 'wss://api.mcmf-system.com/ws',
  timeout: 10000,
  retryAttempts: 3
};

// Main App Component
const MCMFMobileApp = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [userToken, setUserToken] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('connected');

  useEffect(() => {
    initializeApp();
    setupPushNotifications();
    checkNetworkStatus();
  }, []);

  const initializeApp = async () => {
    try {
      const token = await AsyncStorage.getItem('authToken');
      if (token) {
        setUserToken(token);
        setIsAuthenticated(true);
      }
    } catch (error) {
      console.error('App initialization error:', error);
    }
  };

  const setupPushNotifications = () => {
    PushNotification.configure({
      onNotification: function(notification) {
        console.log('NOTIFICATION:', notification);
        if (notification.userInteraction) {
          // Handle notification tap
          handleNotificationTap(notification);
        }
      },
      requestPermissions: Platform.OS === 'ios',
    });

    // Create notification channels for Android
    PushNotification.createChannel(
      {
        channelId: 'risk-alerts',
        channelName: 'Risk Alerts',
        channelDescription: 'Critical risk alerts and portfolio notifications',
        playSound: true,
        soundName: 'default',
        importance: 4,
        vibrate: true,
      },
      (created) => console.log(`Risk alerts channel created: ${created}`)
    );
  };

  const checkNetworkStatus = () => {
    const unsubscribe = NetInfo.addEventListener(state => {
      setConnectionStatus(state.isConnected ? 'connected' : 'offline');
      if (!state.isConnected) {
        Alert.alert('Connection Lost', 'App is now in offline mode. Some features may be limited.');
      }
    });

    return unsubscribe;
  };

  const handleNotificationTap = (notification) => {
    // Navigate to relevant screen based on notification type
    const { type, data } = notification;
    switch (type) {
      case 'risk_alert':
        // Navigate to risk dashboard
        break;
      case 'portfolio_update':
        // Navigate to portfolio screen
        break;
      default:
        break;
    }
  };

  return (
    <NavigationContainer>
      <StatusBar barStyle="dark-content" backgroundColor={theme.colors.light} />
      <SafeAreaView style={styles.container}>
        {isAuthenticated ? (
          <MainTabNavigator connectionStatus={connectionStatus} userToken={userToken} />
        ) : (
          <AuthScreen onAuthentication={setIsAuthenticated} onTokenReceived={setUserToken} />
        )}
      </SafeAreaView>
    </NavigationContainer>
  );
};

// Authentication Screen
const AuthScreen = ({ onAuthentication, onTokenReceived }) => {
  const [loading, setLoading] = useState(false);
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 1000,
      useNativeDriver: true,
    }).start();
  }, []);

  const handleLogin = async (credentials) => {
    setLoading(true);
    try {
      // Mock authentication - replace with real API call
      const response = await fetch(`${API_CONFIG.baseURL}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(credentials),
      });

      const data = await response.json();
      
      if (data.token) {
        await AsyncStorage.setItem('authToken', data.token);
        onTokenReceived(data.token);
        onAuthentication(true);
      }
    } catch (error) {
      Alert.alert('Authentication Error', 'Please check your credentials and try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <LinearGradient
      colors={[theme.colors.primary, theme.colors.secondary]}
      style={styles.authContainer}
    >
      <Animated.View style={[styles.authCard, { opacity: fadeAnim }]}>
        <Icon name="finance" size={60} color={theme.colors.primary} style={styles.authIcon} />
        <Text style={styles.authTitle}>MCMF Mobile</Text>
        <Text style={styles.authSubtitle}>Advanced Portfolio Analytics</Text>
        
        <TouchableOpacity 
          style={styles.loginButton}
          onPress={() => handleLogin({ username: 'demo', password: 'demo123' })}
          disabled={loading}
        >
          <Text style={styles.loginButtonText}>
            {loading ? 'Authenticating...' : 'Login with Demo Account'}
          </Text>
        </TouchableOpacity>

        <Text style={styles.authFooter}>
          Secure • Real-time • Professional
        </Text>
      </Animated.View>
    </LinearGradient>
  );
};

// Main Tab Navigator
const MainTabNavigator = ({ connectionStatus, userToken }) => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color, size }) => {
          let iconName;
          switch (route.name) {
            case 'Dashboard':
              iconName = 'view-dashboard';
              break;
            case 'Portfolio':
              iconName = 'briefcase';
              break;
            case 'Risk':
              iconName = 'shield-alert';
              break;
            case 'Analytics':
              iconName = 'chart-line';
              break;
            case 'Settings':
              iconName = 'cog';
              break;
          }
          return <Icon name={iconName} size={size} color={color} />;
        },
        tabBarActiveTintColor: theme.colors.primary,
        tabBarInactiveTintColor: theme.colors.textSecondary,
        tabBarStyle: {
          backgroundColor: theme.colors.background,
          borderTopColor: theme.colors.surface,
          height: 60,
          paddingBottom: 8,
        },
        headerShown: false,
      })}
    >
      <Tab.Screen name="Dashboard">
        {props => <DashboardScreen {...props} connectionStatus={connectionStatus} userToken={userToken} />}
      </Tab.Screen>
      <Tab.Screen name="Portfolio">
        {props => <PortfolioScreen {...props} userToken={userToken} />}
      </Tab.Screen>
      <Tab.Screen name="Risk">
        {props => <RiskScreen {...props} userToken={userToken} />}
      </Tab.Screen>
      <Tab.Screen name="Analytics">
        {props => <AnalyticsScreen {...props} userToken={userToken} />}
      </Tab.Screen>
      <Tab.Screen name="Settings">
        {props => <SettingsScreen {...props} userToken={userToken} />}
      </Tab.Screen>
    </Tab.Navigator>
  );
};

// Dashboard Screen
const DashboardScreen = ({ connectionStatus, userToken }) => {
  const [portfolioData, setPortfolioData] = useState(null);
  const [marketData, setMarketData] = useState({});
  const [riskMetrics, setRiskMetrics] = useState({});
  const [refreshing, setRefreshing] = useState(false);
  const [websocket, setWebsocket] = useState(null);

  useEffect(() => {
    loadDashboardData();
    setupWebSocket();
    
    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  const setupWebSocket = () => {
    if (connectionStatus === 'offline') return;

    const ws = new WebSocket(`${API_CONFIG.websocketURL}?token=${userToken}`);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWebsocket(ws);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleRealtimeUpdate(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      // Attempt to reconnect after 5 seconds
      setTimeout(setupWebSocket, 5000);
    };
  };

  const handleRealtimeUpdate = (data) => {
    switch (data.type) {
      case 'portfolio_update':
        setPortfolioData(prev => ({ ...prev, ...data.payload }));
        break;
      case 'market_update':
        setMarketData(prev => ({ ...prev, ...data.payload }));
        break;
      case 'risk_alert':
        showRiskAlert(data.payload);
        break;
    }
  };

  const showRiskAlert = (alertData) => {
    PushNotification.localNotification({
      channelId: 'risk-alerts',
      title: 'Risk Alert',
      message: alertData.message,
      playSound: true,
      soundName: 'default',
      actions: ['View Details'],
    });
  };

  const loadDashboardData = async () => {
    try {
      const [portfolio, market, risk] = await Promise.all([
        fetchPortfolioSummary(),
        fetchMarketOverview(),
        fetchRiskMetrics()
      ]);

      setPortfolioData(portfolio);
      setMarketData(market);
      setRiskMetrics(risk);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      if (connectionStatus === 'offline') {
        loadOfflineData();
      }
    }
  };

  const loadOfflineData = async () => {
    try {
      const cachedData = await AsyncStorage.getItem('dashboardCache');
      if (cachedData) {
        const { portfolio, market, risk } = JSON.parse(cachedData);
        setPortfolioData(portfolio);
        setMarketData(market);
        setRiskMetrics(risk);
      }
    } catch (error) {
      console.error('Error loading offline data:', error);
    }
  };

  const fetchPortfolioSummary = async () => {
    // Mock data - replace with actual API call
    return {
      totalValue: 1250000,
      dailyPnL: 15420,
      dailyPnLPercent: 1.25,
      positions: [
        { symbol: 'AAPL', value: 312500, weight: 0.25, pnl: 5200 },
        { symbol: 'MSFT', value: 250000, weight: 0.20, pnl: 3800 },
        { symbol: 'GOOGL', value: 187500, weight: 0.15, pnl: 2100 },
        { symbol: 'TSLA', value: 250000, weight: 0.20, pnl: 4320 },
        { symbol: 'BTC', value: 250000, weight: 0.20, pnl: 8200 }
      ]
    };
  };

  const fetchMarketOverview = async () => {
    return {
      indices: {
        'S&P 500': { value: 4450.12, change: 1.23, changePercent: 0.28 },
        'NASDAQ': { value: 13850.45, change: 89.34, changePercent: 0.65 },
        'Bitcoin': { value: 67420.50, change: 2156.78, changePercent: 3.31 }
      },
      currencies: {
        'EUR/USD': { value: 1.0856, change: -0.0023, changePercent: -0.21 },
        'GBP/USD': { value: 1.2645, change: 0.0034, changePercent: 0.27 }
      }
    };
  };

  const fetchRiskMetrics = async () => {
    return {
      var95: -0.0234,
      expectedShortfall: -0.0312,
      maxDrawdown: -0.0845,
      sharpeRatio: 1.456,
      volatility: 0.186,
      beta: 1.12
    };
  };

  const onRefresh = async () => {
    setRefreshing(true);
    await loadDashboardData();
    setRefreshing(false);
  };

  if (!portfolioData) {
    return (
      <View style={styles.loadingContainer}>
        <Icon name="loading" size={40} color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading Dashboard...</Text>
      </View>
    );
  }

  return (
    <ScrollView 
      style={styles.container}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} />}
    >
      {/* Connection Status Banner */}
      {connectionStatus === 'offline' && (
        <View style={styles.offlineBanner}>
          <Icon name="wifi-off" size={16} color={theme.colors.light} />
          <Text style={styles.offlineBannerText}>Offline Mode - Showing Cached Data</Text>
        </View>
      )}

      {/* Portfolio Summary Card */}
      <DashboardCard title="Portfolio Overview">
        <View style={styles.portfolioSummary}>
          <View style={styles.summaryItem}>
            <Text style={styles.summaryValue}>
              ${portfolioData.totalValue.toLocaleString()}
            </Text>
            <Text style={styles.summaryLabel}>Total Value</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={[
              styles.summaryValue,
              { color: portfolioData.dailyPnL >= 0 ? theme.colors.success : theme.colors.danger }
            ]}>
              {portfolioData.dailyPnL >= 0 ? '+' : ''}${portfolioData.dailyPnL.toLocaleString()}
            </Text>
            <Text style={styles.summaryLabel}>Daily P&L</Text>
          </View>
          <View style={styles.summaryItem}>
            <Text style={[
              styles.summaryValue,
              { color: portfolioData.dailyPnLPercent >= 0 ? theme.colors.success : theme.colors.danger }
            ]}>
              {portfolioData.dailyPnLPercent >= 0 ? '+' : ''}{portfolioData.dailyPnLPercent.toFixed(2)}%
            </Text>
            <Text style={styles.summaryLabel}>Daily Return</Text>
          </View>
        </View>
      </DashboardCard>

      {/* Portfolio Allocation Chart */}
      <DashboardCard title="Asset Allocation">
        <PieChart
          data={portfolioData.positions.map(pos => ({
            name: pos.symbol,
            value: pos.value,
            color: getRandomColor(),
            legendFontColor: theme.colors.text,
            legendFontSize: 12
          }))}
          width={width - 64}
          height={200}
          chartConfig={{
            color: (opacity = 1) => `rgba(${theme.colors.primary}, ${opacity})`,
          }}
          accessor="value"
          backgroundColor="transparent"
          paddingLeft="15"
          absolute
        />
      </DashboardCard>

      {/* Risk Metrics */}
      <DashboardCard title="Risk Metrics">
        <View style={styles.riskMetricsGrid}>
          <RiskMetricItem 
            label="VaR (95%)" 
            value={`${(riskMetrics.var95 * 100).toFixed(2)}%`}
            color={theme.colors.danger}
            icon="alert-circle"
          />
          <RiskMetricItem 
            label="Sharpe Ratio" 
            value={riskMetrics.sharpeRatio.toFixed(3)}
            color={theme.colors.success}
            icon="trending-up"
          />
          <RiskMetricItem 
            label="Max Drawdown" 
            value={`${(riskMetrics.maxDrawdown * 100).toFixed(2)}%`}
            color={theme.colors.warning}
            icon="trending-down"
          />
          <RiskMetricItem 
            label="Volatility" 
            value={`${(riskMetrics.volatility * 100).toFixed(1)}%`}
            color={theme.colors.info}
            icon="chart-line-variant"
          />
        </View>
      </DashboardCard>

      {/* Market Overview */}
      <DashboardCard title="Market Overview">
        {Object.entries(marketData.indices || {}).map(([name, data]) => (
          <MarketIndexItem 
            key={name}
            name={name}
            value={data.value}
            change={data.change}
            changePercent={data.changePercent}
          />
        ))}
      </DashboardCard>

      {/* Quick Actions */}
      <DashboardCard title="Quick Actions">
        <View style={styles.quickActionsGrid}>
          <QuickActionButton 
            icon="chart-timeline-variant"
            label="Run Simulation"
            onPress={() => {}}
          />
          <QuickActionButton 
            icon="shield-check"
            label="Stress Test"
            onPress={() => {}}
          />
          <QuickActionButton 
            icon="leaf"
            label="ESG Analysis"
            onPress={() => {}}
          />
          <QuickActionButton 
            icon="bitcoin"
            label="Crypto Monitor"
            onPress={() => {}}
          />
        </View>
      </DashboardCard>
    </ScrollView>
  );
};

// Dashboard Card Component
const DashboardCard = ({ title, children }) => (
  <View style={styles.dashboardCard}>
    <Text style={styles.cardTitle}>{title}</Text>
    {children}
  </View>
);

// Risk Metric Item Component
const RiskMetricItem = ({ label, value, color, icon }) => (
  <View style={styles.riskMetricItem}>
    <Icon name={icon} size={24} color={color} />
    <Text style={[styles.riskMetricValue, { color }]}>{value}</Text>
    <Text style={styles.riskMetricLabel}>{label}</Text>
  </View>
);

// Market Index Item Component
const MarketIndexItem = ({ name, value, change, changePercent }) => (
  <View style={styles.marketIndexItem}>
    <View style={styles.marketIndexInfo}>
      <Text style={styles.marketIndexName}>{name}</Text>
      <Text style={styles.marketIndexValue}>{value.toLocaleString()}</Text>
    </View>
    <View style={styles.marketIndexChange}>
      <Text style={[
        styles.marketIndexChangeValue,
        { color: change >= 0 ? theme.colors.success : theme.colors.danger }
      ]}>
        {change >= 0 ? '+' : ''}{change.toFixed(2)}
      </Text>
      <Text style={[
        styles.marketIndexChangePercent,
        { color: changePercent >= 0 ? theme.colors.success : theme.colors.danger }
      ]}>
        ({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
      </Text>
    </View>
  </View>
);

// Quick Action Button Component
const QuickActionButton = ({ icon, label, onPress }) => (
  <TouchableOpacity style={styles.quickActionButton} onPress={onPress}>
    <Icon name={icon} size={32} color={theme.colors.primary} />
    <Text style={styles.quickActionLabel}>{label}</Text>
  </TouchableOpacity>
);

// Portfolio Screen
const PortfolioScreen = ({ userToken }) => {
  const [positions, setPositions] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
  const [selectedTimeframe, setSelectedTimeframe] = useState('1D');

  useEffect(() => {
    loadPortfolioData();
  }, [selectedTimeframe]);

  const loadPortfolioData = async () => {
    // Mock portfolio data
    setPositions([
      { 
        symbol: 'AAPL', 
        name: 'Apple Inc.', 
        shares: 100, 
        avgCost: 150.00, 
        currentPrice: 175.20, 
        marketValue: 17520,
        unrealizedPL: 2520,
        unrealizedPLPercent: 16.8,
        dayChange: 2.35,
        dayChangePercent: 1.36,
        sector: 'Technology',
        esgScore: 8.2,
        allocation: 25.0
      },
      {
        symbol: 'TSLA',
        name: 'Tesla Inc.',
        shares: 50,
        avgCost: 200.00,
        currentPrice: 245.80,
        marketValue: 12290,
        unrealizedPL: 2290,
        unrealizedPLPercent: 22.9,
        dayChange: 8.45,
        dayChangePercent: 3.56,
        sector: 'Automotive',
        esgScore: 7.8,
        allocation: 17.5
      }
    ]);

    // Mock performance data
    const timeframes = {
      '1D': generateMockData(24, 'hours'),
      '1W': generateMockData(7, 'days'),
      '1M': generateMockData(30, 'days'),
      '3M': generateMockData(90, 'days'),
      '1Y': generateMockData(365, 'days')
    };

    setPerformanceData(timeframes[selectedTimeframe]);
  };

  const generateMockData = (periods, unit) => {
    const data = [];
    const baseValue = 100000;
    let currentValue = baseValue;

    for (let i = 0; i < periods; i++) {
      const change = (Math.random() - 0.5) * 0.02; // ±1% random change
      currentValue *= (1 + change);
      data.push({
        timestamp: new Date(Date.now() - (periods - i) * (unit === 'hours' ? 3600000 : 86400000)),
        value: currentValue
      });
    }
    return data;
  };

  const getTimeframeColor = (timeframe) => {
    return selectedTimeframe === timeframe ? theme.colors.primary : theme.colors.textSecondary;
  };

  return (
    <ScrollView style={styles.container}>
      {/* Performance Chart */}
      <DashboardCard title="Portfolio Performance">
        {/* Timeframe Selector */}
        <View style={styles.timeframeSelector}>
          {['1D', '1W', '1M', '3M', '1Y'].map(tf => (
            <TouchableOpacity
              key={tf}
              style={[
                styles.timeframeButton,
                selectedTimeframe === tf && styles.timeframeButtonActive
              ]}
              onPress={() => setSelectedTimeframe(tf)}
            >
              <Text style={[
                styles.timeframeButtonText,
                { color: getTimeframeColor(tf) }
              ]}>{tf}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Performance Line Chart */}
        <LineChart
          data={{
            datasets: [{
              data: performanceData.map(d => d.value),
              strokeWidth: 2,
              color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`
            }]
          }}
          width={width - 64}
          height={200}
          chartConfig={{
            backgroundColor: 'transparent',
            backgroundGradientFrom: theme.colors.background,
            backgroundGradientTo: theme.colors.background,
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(108, 117, 125, ${opacity})`,
            style: {
              borderRadius: 16
            },
            propsForDots: {
              r: "0",
            }
          }}
          bezier
          withDots={false}
          withInnerLines={false}
          withOuterLines={false}
          withVerticalLines={false}
          withHorizontalLines={true}
        />
      </DashboardCard>

      {/* Holdings List */}
      <DashboardCard title="Holdings">
        {positions.map((position, index) => (
          <PositionItem key={index} position={position} />
        ))}
      </DashboardCard>

      {/* Sector Allocation */}
      <DashboardCard title="Sector Allocation">
        <BarChart
          data={{
            labels: ['Tech', 'Finance', 'Healthcare', 'Energy', 'Auto'],
            datasets: [{
              data: [35, 20, 15, 15, 15]
            }]
          }}
          width={width - 64}
          height={200}
          chartConfig={{
            backgroundColor: 'transparent',
            backgroundGradientFrom: theme.colors.background,
            backgroundGradientTo: theme.colors.background,
            decimalPlaces: 0,
            color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(108, 117, 125, ${opacity})`,
          }}
          verticalLabelRotation={30}
        />
      </DashboardCard>
    </ScrollView>
  );
};

// Position Item Component
const PositionItem = ({ position }) => (
  <TouchableOpacity style={styles.positionItem}>
    <View style={styles.positionHeader}>
      <View>
        <Text style={styles.positionSymbol}>{position.symbol}</Text>
        <Text style={styles.positionName}>{position.name}</Text>
      </View>
      <View style={styles.positionAllocation}>
        <Text style={styles.allocationText}>{position.allocation.toFixed(1)}%</Text>
        <View style={styles.esgBadge}>
          <Text style={styles.esgScore}>ESG {position.esgScore}</Text>
        </View>
      </View>
    </View>

    <View style={styles.positionDetails}>
      <View style={styles.positionMetric}>
        <Text style={styles.metricLabel}>Shares</Text>
        <Text style={styles.metricValue}>{position.shares}</Text>
      </View>
      <View style={styles.positionMetric}>
        <Text style={styles.metricLabel}>Avg Cost</Text>
        <Text style={styles.metricValue}>${position.avgCost.toFixed(2)}</Text>
      </View>
      <View style={styles.positionMetric}>
        <Text style={styles.metricLabel}>Current</Text>
        <Text style={styles.metricValue}>${position.currentPrice.toFixed(2)}</Text>
      </View>
      <View style={styles.positionMetric}>
        <Text style={styles.metricLabel}>Market Value</Text>
        <Text style={styles.metricValue}>${position.marketValue.toLocaleString()}</Text>
      </View>
    </View>

    <View style={styles.positionPnL}>
      <View style={styles.unrealizedPnL}>
        <Text style={[
          styles.pnlValue,
          { color: position.unrealizedPL >= 0 ? theme.colors.success : theme.colors.danger }
        ]}>
          {position.unrealizedPL >= 0 ? '+' : ''}${position.unrealizedPL.toLocaleString()}
        </Text>
        <Text style={[
          styles.pnlPercent,
          { color: position.unrealizedPLPercent >= 0 ? theme.colors.success : theme.colors.danger }
        ]}>
          ({position.unrealizedPLPercent >= 0 ? '+' : ''}{position.unrealizedPLPercent.toFixed(1)}%)
        </Text>
      </View>
      <View style={styles.dayChange}>
        <Text style={[
          styles.dayChangeValue,
          { color: position.dayChange >= 0 ? theme.colors.success : theme.colors.danger }
        ]}>
          {position.dayChange >= 0 ? '+' : ''}${position.dayChange.toFixed(2)}
        </Text>
        <Text style={[
          styles.dayChangePercent,
          { color: position.dayChangePercent >= 0 ? theme.colors.success : theme.colors.danger }
        ]}>
          ({position.dayChangePercent >= 0 ? '+' : ''}{position.dayChangePercent.toFixed(2)}%)
        </Text>
      </View>
    </View>
  </TouchableOpacity>
);

// Risk Screen
const RiskScreen = ({ userToken }) => {
  const [riskData, setRiskData] = useState(null);
  const [stressTestResults, setStressTestResults] = useState([]);
  const [selectedRiskTimeframe, setSelectedRiskTimeframe] = useState('1M');

  useEffect(() => {
    loadRiskData();
  }, [selectedRiskTimeframe]);

  const loadRiskData = async () => {
    // Mock risk data
    setRiskData({
      var95: -0.0234,
      var99: -0.0456,
      expectedShortfall95: -0.0312,
      expectedShortfall99: -0.0567,
      maxDrawdown: -0.0845,
      sharpeRatio: 1.456,
      sortinoRatio: 1.789,
      calmarRatio: 2.134,
      volatility: 0.186,
      beta: 1.12,
      tracking_error: 0.045,
      information_ratio: 0.234
    });

    // Mock stress test results
    setStressTestResults([
      { scenario: '2008 Financial Crisis', impact: -0.38, probability: 0.05 },
      { scenario: 'COVID-19 Crash', impact: -0.34, probability: 0.02 },
      { scenario: 'Tech Bubble Burst', impact: -0.29, probability: 0.08 },
      { scenario: 'Interest Rate Shock', impact: -0.15, probability: 0.15 },
      { scenario: 'Geopolitical Crisis', impact: -0.22, probability: 0.10 }
    ]);
  };

  if (!riskData) {
    return (
      <View style={styles.loadingContainer}>
        <Icon name="loading" size={40} color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading Risk Analytics...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* VaR Analysis */}
      <DashboardCard title="Value at Risk Analysis">
        <View style={styles.varAnalysis}>
          <View style={styles.varItem}>
            <Text style={styles.varLabel}>95% VaR</Text>
            <Text style={[styles.varValue, { color: theme.colors.danger }]}>
              {(riskData.var95 * 100).toFixed(2)}%
            </Text>
          </View>
          <View style={styles.varItem}>
            <Text style={styles.varLabel}>99% VaR</Text>
            <Text style={[styles.varValue, { color: theme.colors.danger }]}>
              {(riskData.var99 * 100).toFixed(2)}%
            </Text>
          </View>
          <View style={styles.varItem}>
            <Text style={styles.varLabel}>Expected Shortfall 95%</Text>
            <Text style={[styles.varValue, { color: theme.colors.danger }]}>
              {(riskData.expectedShortfall95 * 100).toFixed(2)}%
            </Text>
          </View>
          <View style={styles.varItem}>
            <Text style={styles.varLabel}>Expected Shortfall 99%</Text>
            <Text style={[styles.varValue, { color: theme.colors.danger }]}>
              {(riskData.expectedShortfall99 * 100).toFixed(2)}%
            </Text>
          </View>
        </View>
      </DashboardCard>

      {/* Risk-Adjusted Returns */}
      <DashboardCard title="Risk-Adjusted Performance">
        <View style={styles.riskAdjustedGrid}>
          <RiskMetricItem 
            label="Sharpe Ratio" 
            value={riskData.sharpeRatio.toFixed(3)}
            color={theme.colors.success}
            icon="trending-up"
          />
          <RiskMetricItem 
            label="Sortino Ratio" 
            value={riskData.sortinoRatio.toFixed(3)}
            color={theme.colors.success}
            icon="chart-line"
          />
          <RiskMetricItem 
            label="Calmar Ratio" 
            value={riskData.calmarRatio.toFixed(3)}
            color={theme.colors.info}
            icon="chart-timeline-variant"
          />
          <RiskMetricItem 
            label="Information Ratio" 
            value={riskData.information_ratio.toFixed(3)}
            color={theme.colors.warning}
            icon="information"
          />
        </View>
      </DashboardCard>

      {/* Stress Test Results */}
      <DashboardCard title="Stress Test Results">
        {stressTestResults.map((test, index) => (
          <View key={index} style={styles.stressTestItem}>
            <View style={styles.stressTestInfo}>
              <Text style={styles.stressTestScenario}>{test.scenario}</Text>
              <Text style={styles.stressTestProbability}>
                Probability: {(test.probability * 100).toFixed(1)}%
              </Text>
            </View>
            <View style={styles.stressTestImpact}>
              <Text style={[
                styles.stressTestImpactValue,
                { color: theme.colors.danger }
              ]}>
                {(test.impact * 100).toFixed(1)}%
              </Text>
              <View style={styles.stressTestBar}>
                <View 
                  style={[
                    styles.stressTestBarFill,
                    { width: `${Math.abs(test.impact) * 100}%` }
                  ]}
                />
              </View>
            </View>
          </View>
        ))}
      </DashboardCard>

      {/* Risk Alerts */}
      <DashboardCard title="Risk Alerts">
        <View style={styles.riskAlert}>
          <Icon name="alert-circle" size={20} color={theme.colors.warning} />
          <View style={styles.riskAlertContent}>
            <Text style={styles.riskAlertTitle}>Portfolio Concentration Alert</Text>
            <Text style={styles.riskAlertMessage}>
              Technology sector allocation (35%) exceeds recommended limit of 30%
            </Text>
          </View>
        </View>
        
        <View style={styles.riskAlert}>
          <Icon name="shield-check" size={20} color={theme.colors.success} />
          <View style={styles.riskAlertContent}>
            <Text style={styles.riskAlertTitle}>Risk Limits Compliant</Text>
            <Text style={styles.riskAlertMessage}>
              All risk metrics are within established limits
            </Text>
          </View>
        </View>
      </DashboardCard>
    </ScrollView>
  );
};

// Analytics Screen
const AnalyticsScreen = ({ userToken }) => {
  const [analyticsData, setAnalyticsData] = useState(null);
  const [selectedAnalytics, setSelectedAnalytics] = useState('performance');

  useEffect(() => {
    loadAnalyticsData();
  }, [selectedAnalytics]);

  const loadAnalyticsData = async () => {
    // Mock analytics data
    setAnalyticsData({
      performance: {
        monthlyReturns: [2.3, -1.2, 4.5, 1.8, -0.5, 3.2, 2.1, -2.3, 1.9, 0.8, 2.7, 1.4],
        benchmark: [1.8, -0.8, 3.9, 1.5, -0.3, 2.8, 1.9, -1.8, 1.6, 0.5, 2.3, 1.2],
        alpha: 0.045,
        beta: 1.12,
        r_squared: 0.87
      },
      attribution: {
        sectors: [
          { name: 'Technology', contribution: 1.2 },
          { name: 'Healthcare', contribution: 0.8 },
          { name: 'Financial', contribution: 0.3 },
          { name: 'Energy', contribution: -0.2 },
          { name: 'Utilities', contribution: 0.1 }
        ],
        factors: [
          { name: 'Stock Selection', contribution: 0.8 },
          { name: 'Asset Allocation', contribution: 0.6 },
          { name: 'Market Timing', contribution: -0.2 }
        ]
      }
    });
  };

  const renderPerformanceAnalytics = () => (
    <>
      <DashboardCard title="Performance vs Benchmark">
        <LineChart
          data={{
            labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            datasets: [
              {
                data: analyticsData.performance.monthlyReturns.slice(0, 6),
                color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
                strokeWidth: 2
              },
              {
                data: analyticsData.performance.benchmark.slice(0, 6),
                color: (opacity = 1) => `rgba(118, 75, 162, ${opacity})`,
                strokeWidth: 2
              }
            ]
          }}
          width={width - 64}
          height={200}
          chartConfig={{
            backgroundColor: 'transparent',
            backgroundGradientFrom: theme.colors.background,
            backgroundGradientTo: theme.colors.background,
            decimalPlaces: 1,
            color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(108, 117, 125, ${opacity})`,
          }}
          bezier
        />
        
        <View style={styles.performanceMetrics}>
          <View style={styles.performanceMetric}>
            <Text style={styles.performanceMetricLabel}>Alpha</Text>
            <Text style={styles.performanceMetricValue}>
              {(analyticsData.performance.alpha * 100).toFixed(2)}%
            </Text>
          </View>
          <View style={styles.performanceMetric}>
            <Text style={styles.performanceMetricLabel}>Beta</Text>
            <Text style={styles.performanceMetricValue}>
              {analyticsData.performance.beta.toFixed(2)}
            </Text>
          </View>
          <View style={styles.performanceMetric}>
            <Text style={styles.performanceMetricLabel}>R²</Text>
            <Text style={styles.performanceMetricValue}>
              {analyticsData.performance.r_squared.toFixed(2)}
            </Text>
          </View>
        </View>
      </DashboardCard>
    </>
  );

  const renderAttributionAnalytics = () => (
    <>
      <DashboardCard title="Sector Attribution">
        <BarChart
          data={{
            labels: analyticsData.attribution.sectors.map(s => s.name.substring(0, 4)),
            datasets: [{
              data: analyticsData.attribution.sectors.map(s => s.contribution)
            }]
          }}
          width={width - 64}
          height={200}
          chartConfig={{
            backgroundColor: 'transparent',
            backgroundGradientFrom: theme.colors.background,
            backgroundGradientTo: theme.colors.background,
            decimalPlaces: 1,
            color: (opacity = 1) => `rgba(102, 126, 234, ${opacity})`,
            labelColor: (opacity = 1) => `rgba(108, 117, 125, ${opacity})`,
          }}
        />
      </DashboardCard>

      <DashboardCard title="Factor Attribution">
        {analyticsData.attribution.factors.map((factor, index) => (
          <View key={index} style={styles.factorAttributionItem}>
            <Text style={styles.factorName}>{factor.name}</Text>
            <View style={styles.factorContribution}>
              <View style={[
                styles.factorBar,
                { 
                  width: `${Math.abs(factor.contribution) * 50}%`,
                  backgroundColor: factor.contribution >= 0 ? theme.colors.success : theme.colors.danger
                }
              ]} />
              <Text style={[
                styles.factorContributionText,
                { color: factor.contribution >= 0 ? theme.colors.success : theme.colors.danger }
              ]}>
                {factor.contribution >= 0 ? '+' : ''}{factor.contribution.toFixed(2)}%
              </Text>
            </View>
          </View>
        ))}
      </DashboardCard>
    </>
  );

  if (!analyticsData) {
    return (
      <View style={styles.loadingContainer}>
        <Icon name="loading" size={40} color={theme.colors.primary} />
        <Text style={styles.loadingText}>Loading Analytics...</Text>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      {/* Analytics Type Selector */}
      <View style={styles.analyticsSelector}>
        <TouchableOpacity
          style={[
            styles.analyticsSelectorButton,
            selectedAnalytics === 'performance' && styles.analyticsSelectorButtonActive
          ]}
          onPress={() => setSelectedAnalytics('performance')}
        >
          <Text style={[
            styles.analyticsSelectorText,
            selectedAnalytics === 'performance' && styles.analyticsSelectorTextActive
          ]}>Performance</Text>
        </TouchableOpacity>
        
        <TouchableOpacity
          style={[
            styles.analyticsSelectorButton,
            selectedAnalytics === 'attribution' && styles.analyticsSelectorButtonActive
          ]}
          onPress={() => setSelectedAnalytics('attribution')}
        >
          <Text style={[
            styles.analyticsSelectorText,
            selectedAnalytics === 'attribution' && styles.analyticsSelectorTextActive
          ]}>Attribution</Text>
        </TouchableOpacity>
      </View>

      {selectedAnalytics === 'performance' && renderPerformanceAnalytics()}
      {selectedAnalytics === 'attribution' && renderAttributionAnalytics()}
    </ScrollView>
  );
};

// Settings Screen
const SettingsScreen = ({ userToken }) => {
  const [notifications, setNotifications] = useState(true);
  const [biometric, setBiometric] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  const handleLogout = async () => {
    Alert.alert(
      'Logout',
      'Are you sure you want to logout?',
      [
        { text: 'Cancel', style: 'cancel' },
        { 
          text: 'Logout', 
          style: 'destructive',
          onPress: async () => {
            await AsyncStorage.removeItem('authToken');
            // Navigate to login screen or reset app state
          }
        }
      ]
    );
  };

  return (
    <ScrollView style={styles.container}>
      <DashboardCard title="Notifications">
        <SettingsItem
          icon="bell"
          label="Push Notifications"
          value={notifications}
          onToggle={setNotifications}
          type="switch"
        />
        <SettingsItem
          icon="shield-alert"
          label="Risk Alerts"
          value={true}
          type="switch"
        />
        <SettingsItem
          icon="chart-line"
          label="Portfolio Updates"
          value={true}
          type="switch"
        />
      </DashboardCard>

      <DashboardCard title="Security">
        <SettingsItem
          icon="fingerprint"
          label="Biometric Authentication"
          value={biometric}
          onToggle={setBiometric}
          type="switch"
        />
        <SettingsItem
          icon="lock"
          label="Change Password"
          type="action"
          onPress={() => {}}
        />
        <SettingsItem
          icon="two-factor-authentication"
          label="Two-Factor Authentication"
          type="action"
          onPress={() => {}}
        />
      </DashboardCard>

      <DashboardCard title="Preferences">
        <SettingsItem
          icon="theme-light-dark"
          label="Dark Mode"
          value={darkMode}
          onToggle={setDarkMode}
          type="switch"
        />
        <SettingsItem
          icon="currency-usd"
          label="Base Currency"
          value="USD"
          type="action"
          onPress={() => {}}
        />
        <SettingsItem
          icon="clock"
          label="Market Hours"
          value="EST"
          type="action"
          onPress={() => {}}
        />
      </DashboardCard>

      <DashboardCard title="Account">
        <SettingsItem
          icon="account"
          label="Profile"
          type="action"
          onPress={() => {}}
        />
        <SettingsItem
          icon="help-circle"
          label="Help & Support"
          type="action"
          onPress={() => {}}
        />
        <SettingsItem
          icon="information"
          label="About"
          type="action"
          onPress={() => {}}
        />
        <SettingsItem
          icon="logout"
          label="Logout"
          type="action"
          onPress={handleLogout}
          textColor={theme.colors.danger}
        />
      </DashboardCard>
    </ScrollView>
  );
};

// Settings Item Component
const SettingsItem = ({ icon, label, value, onToggle, type, onPress, textColor }) => (
  <TouchableOpacity
    style={styles.settingsItem}
    onPress={type === 'action' ? onPress : undefined}
    disabled={type === 'switch'}
  >
    <View style={styles.settingsItemLeft}>
      <Icon name={icon} size={24} color={textColor || theme.colors.text} />
      <Text style={[styles.settingsItemLabel, { color: textColor || theme.colors.text }]}>
        {label}
      </Text>
    </View>
    <View style={styles.settingsItemRight}>
      {type === 'switch' ? (
        <Switch
          value={value}
          onValueChange={onToggle}
          trackColor={{ false: theme.colors.surface, true: theme.colors.primary }}
          thumbColor={value ? theme.colors.light : theme.colors.textSecondary}
        />
      ) : type === 'action' ? (
        <Icon name="chevron-right" size={20} color={theme.colors.textSecondary} />
      ) : (
        <Text style={styles.settingsItemValue}>{value}</Text>
      )}
    </View>
  </TouchableOpacity>
);

// Utility Functions
const getRandomColor = () => {
  const colors = [
    '#FF6384',
    '#36A2EB',
    '#FFCE56',
    '#4BC0C0',
    '#9966FF',
    '#FF9F40'
  ];
  return colors[Math.floor(Math.random() * colors.length)];
};

// Styles
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: theme.colors.light,
  },
  authContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: theme.spacing.lg,
  },
  authCard: {
    backgroundColor: theme.colors.background,
    padding: theme.spacing.xl,
    borderRadius: theme.borderRadius * 2,
    alignItems: 'center',
    width: '100%',
    maxWidth: 300,
    elevation: theme.elevation,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
  },
  authIcon: {
    marginBottom: theme.spacing.lg,
  },
  authTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginBottom: theme.spacing.sm,
  },
  authSubtitle: {
    fontSize: 16,
    color: theme.colors.textSecondary,
    marginBottom: theme.spacing.xl,
    textAlign: 'center',
  },
  loginButton: {
    backgroundColor: theme.colors.primary,
    paddingHorizontal: theme.spacing.lg,
    paddingVertical: theme.spacing.md,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.lg,
    width: '100%',
  },
  loginButtonText: {
    color: theme.colors.light,
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
  },
  authFooter: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    textAlign: 'center',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: theme.spacing.md,
    fontSize: 16,
    color: theme.colors.textSecondary,
  },
  offlineBanner: {
    backgroundColor: theme.colors.warning,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.sm,
  },
  offlineBannerText: {
    color: theme.colors.light,
    marginLeft: theme.spacing.sm,
    fontSize: 14,
    fontWeight: '600',
  },
  dashboardCard: {
    backgroundColor: theme.colors.background,
    marginHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.md,
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius,
    elevation: theme.elevation,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: theme.colors.text,
    marginBottom: theme.spacing.md,
  },
  portfolioSummary: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  summaryItem: {
    alignItems: 'center',
  },
  summaryValue: {
    fontSize: 20,
    fontWeight: 'bold',
    color: theme.colors.text,
  },
  summaryLabel: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.xs,
  },
  riskMetricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  riskMetricItem: {
    width: '48%',
    alignItems: 'center',
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.sm,
  },
  riskMetricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: theme.spacing.xs,
  },
  riskMetricLabel: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: theme.spacing.xs,
    textAlign: 'center',
  },
  marketIndexItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surface,
  },
  marketIndexInfo: {
    flex: 1,
  },
  marketIndexName: {
    fontSize: 14,
    color: theme.colors.text,
    fontWeight: '500',
  },
  marketIndexValue: {
    fontSize: 16,
    color: theme.colors.text,
    fontWeight: 'bold',
    marginTop: 2,
  },
  marketIndexChange: {
    alignItems: 'flex-end',
  },
  marketIndexChangeValue: {
    fontSize: 14,
    fontWeight: '600',
  },
  marketIndexChangePercent: {
    fontSize: 12,
    marginTop: 2,
  },
  quickActionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  quickActionButton: {
    width: '48%',
    alignItems: 'center',
    padding: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.sm,
  },
  quickActionLabel: {
    fontSize: 12,
    color: theme.colors.text,
    marginTop: theme.spacing.xs,
    textAlign: 'center',
  },
  timeframeSelector: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginBottom: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    padding: theme.spacing.xs,
  },
  timeframeButton: {
    paddingVertical: theme.spacing.sm,
    paddingHorizontal: theme.spacing.md,
    borderRadius: theme.borderRadius / 2,
  },
  timeframeButtonActive: {
    backgroundColor: theme.colors.primary,
  },
  timeframeButtonText: {
    fontSize: 14,
    fontWeight: '600',
  },
  positionItem: {
    backgroundColor: theme.colors.surface,
    padding: theme.spacing.md,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.sm,
  },
  positionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing.sm,
  },
  positionSymbol: {
    fontSize: 16,
    fontWeight: 'bold',
    color: theme.colors.text,
  },
  positionName: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: 2,
  },
  positionAllocation: {
    alignItems: 'flex-end',
  },
  allocationText: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.text,
  },
  esgBadge: {
    backgroundColor: theme.colors.success,
    paddingHorizontal: theme.spacing.sm,
    paddingVertical: 2,
    borderRadius: theme.borderRadius / 2,
    marginTop: theme.spacing.xs,
  },
  esgScore: {
    color: theme.colors.light,
    fontSize: 10,
    fontWeight: '600',
  },
  positionDetails: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: theme.spacing.sm,
  },
  positionMetric: {
    alignItems: 'center',
  },
  metricLabel: {
    fontSize: 10,
    color: theme.colors.textSecondary,
  },
  metricValue: {
    fontSize: 12,
    color: theme.colors.text,
    fontWeight: '600',
    marginTop: 2,
  },
  positionPnL: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  unrealizedPnL: {
    alignItems: 'flex-start',
  },
  pnlValue: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  pnlPercent: {
    fontSize: 12,
    marginTop: 2,
  },
  dayChange: {
    alignItems: 'flex-end',
  },
  dayChangeValue: {
    fontSize: 12,
    fontWeight: '600',
  },
  dayChangePercent: {
    fontSize: 10,
    marginTop: 2,
  },
  varAnalysis: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  varItem: {
    width: '48%',
    alignItems: 'center',
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.sm,
  },
  varLabel: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    textAlign: 'center',
  },
  varValue: {
    fontSize: 16,
    fontWeight: 'bold',
    marginTop: theme.spacing.xs,
  },
  riskAdjustedGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  stressTestItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surface,
  },
  stressTestInfo: {
    flex: 1,
  },
  stressTestScenario: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.text,
  },
  stressTestProbability: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: 2,
  },
  stressTestImpact: {
    alignItems: 'flex-end',
    flex: 1,
  },
  stressTestImpactValue: {
    fontSize: 14,
    fontWeight: 'bold',
  },
  stressTestBar: {
    width: 60,
    height: 4,
    backgroundColor: theme.colors.surface,
    borderRadius: 2,
    marginTop: theme.spacing.xs,
  },
  stressTestBarFill: {
    height: '100%',
    backgroundColor: theme.colors.danger,
    borderRadius: 2,
  },
  riskAlert: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: theme.spacing.sm,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    marginBottom: theme.spacing.sm,
  },
  riskAlertContent: {
    flex: 1,
    marginLeft: theme.spacing.sm,
  },
  riskAlertTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.text,
  },
  riskAlertMessage: {
    fontSize: 12,
    color: theme.colors.textSecondary,
    marginTop: 2,
  },
  analyticsSelector: {
    flexDirection: 'row',
    marginHorizontal: theme.spacing.md,
    marginBottom: theme.spacing.md,
    backgroundColor: theme.colors.surface,
    borderRadius: theme.borderRadius,
    padding: theme.spacing.xs,
  },
  analyticsSelectorButton: {
    flex: 1,
    paddingVertical: theme.spacing.sm,
    alignItems: 'center',
    borderRadius: theme.borderRadius / 2,
  },
  analyticsSelectorButtonActive: {
    backgroundColor: theme.colors.primary,
  },
  analyticsSelectorText: {
    fontSize: 14,
    fontWeight: '600',
    color: theme.colors.textSecondary,
  },
  analyticsSelectorTextActive: {
    color: theme.colors.light,
  },
  performanceMetrics: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    marginTop: theme.spacing.md,
  },
  performanceMetric: {
    alignItems: 'center',
  },
  performanceMetricLabel: {
    fontSize: 12,
    color: theme.colors.textSecondary,
  },
  performanceMetricValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: theme.colors.text,
    marginTop: theme.spacing.xs,
  },
  factorAttributionItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surface,
  },
  factorName: {
    fontSize: 14,
    color: theme.colors.text,
    fontWeight: '500',
    flex: 1,
  },
  factorContribution: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
    justifyContent: 'flex-end',
  },
  factorBar: {
    height: 6,
    borderRadius: 3,
    marginRight: theme.spacing.sm,
  },
  factorContributionText: {
    fontSize: 12,
    fontWeight: '600',
    minWidth: 40,
    textAlign: 'right',
  },
  settingsItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: theme.spacing.md,
    borderBottomWidth: 1,
    borderBottomColor: theme.colors.surface,
  },
  settingsItemLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  settingsItemLabel: {
    fontSize: 16,
    color: theme.colors.text,
    marginLeft: theme.spacing.md,
  },
  settingsItemRight: {
    alignItems: 'center',
  },
  settingsItemValue: {
    fontSize: 14,
    color: theme.colors.textSecondary,
  },
});

export default MCMFMobileApp;
