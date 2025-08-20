"""
Interactive dashboards for financial data visualization and monitoring
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import dash
from dash import dcc, html, Input, Output, State
import logging

logger = logging.getLogger(__name__)

class FinanceDashboard:
    """Main financial data dashboard"""
    
    def __init__(self, title: str = "Monte Carlo-Markov Finance Dashboard"):
        self.title = title
        self.figures = {}
        self.data_cache = {}
        
    def create_portfolio_overview(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """Create portfolio overview visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value', 'Daily Returns', 'Drawdown', 'Asset Allocation'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Portfolio value over time
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=portfolio_data['portfolio_value'],
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        if benchmark_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark_data.index,
                    y=benchmark_data['value'],
                    name='Benchmark',
                    line=dict(color='red', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Daily returns
        returns = portfolio_data['portfolio_value'].pct_change().dropna()
        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Daily Returns',
                nbinsx=50,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Drawdown
        peak = portfolio_data['portfolio_value'].expanding().max()
        drawdown = (portfolio_data['portfolio_value'] - peak) / peak
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_data.index,
                y=drawdown,
                fill='tonexty',
                name='Drawdown',
                line=dict(color='red'),
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )
        
        # Asset allocation (if available)
        if 'allocations' in portfolio_data.columns:
            latest_allocation = portfolio_data['allocations'].iloc[-1]
            if isinstance(latest_allocation, dict):
                fig.add_trace(
                    go.Pie(
                        labels=list(latest_allocation.keys()),
                        values=list(latest_allocation.values()),
                        name="Asset Allocation"
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text=self.title,
            showlegend=True
        )
        
        self.figures['portfolio_overview'] = fig
        return fig
        
    def create_risk_metrics_dashboard(
        self,
        risk_data: Dict[str, Any]
    ) -> go.Figure:
        """Create risk metrics dashboard"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Value at Risk', 'Expected Shortfall', 'Risk Contribution',
                'Correlation Heatmap', 'Volatility Surface', 'Stress Tests'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                [{"type": "heatmap"}, {"type": "surface"}, {"type": "scatter"}]
            ]
        )
        
        # VaR at different confidence levels
        if 'var' in risk_data:
            var_data = risk_data['var']
            confidence_levels = list(var_data.keys())
            var_values = list(var_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=confidence_levels,
                    y=var_values,
                    name='VaR',
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # Expected Shortfall
        if 'expected_shortfall' in risk_data:
            es_data = risk_data['expected_shortfall']
            confidence_levels = list(es_data.keys())
            es_values = list(es_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=confidence_levels,
                    y=es_values,
                    name='Expected Shortfall',
                    marker_color='darkred'
                ),
                row=1, col=2
            )
        
        # Risk contribution
        if 'risk_contribution' in risk_data:
            contrib_data = risk_data['risk_contribution']
            assets = list(contrib_data.keys())
            contributions = list(contrib_data.values())
            
            fig.add_trace(
                go.Bar(
                    x=assets,
                    y=contributions,
                    name='Risk Contribution',
                    marker_color='orange'
                ),
                row=1, col=3
            )
        
        # Correlation heatmap
        if 'correlation_matrix' in risk_data:
            corr_matrix = risk_data['correlation_matrix']
            if isinstance(corr_matrix, pd.DataFrame):
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdBu',
                        zmid=0,
                        name='Correlations'
                    ),
                    row=2, col=1
                )
        
        # Stress test results
        if 'stress_tests' in risk_data:
            stress_data = risk_data['stress_tests']
            scenarios = list(stress_data.keys())
            pnl_values = list(stress_data.values())
            
            colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
            
            fig.add_trace(
                go.Scatter(
                    x=scenarios,
                    y=pnl_values,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(width=1, color='black')
                    ),
                    name='Stress Test P&L'
                ),
                row=2, col=3
            )
        
        fig.update_layout(
            height=800,
            title_text="Risk Metrics Dashboard",
            showlegend=True
        )
        
        self.figures['risk_dashboard'] = fig
        return fig
        
    def create_monte_carlo_results(
        self,
        mc_results: Dict[str, Any]
    ) -> go.Figure:
        """Create Monte Carlo simulation results visualization"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Path Simulations', 'Final Value Distribution', 
                'Convergence Analysis', 'Confidence Intervals'
            )
        )
        
        # Sample paths
        if 'paths' in mc_results:
            paths = mc_results['paths']
            n_display = min(100, paths.shape[0])  # Limit displayed paths
            
            for i in range(n_display):
                fig.add_trace(
                    go.Scatter(
                        y=paths[i],
                        mode='lines',
                        line=dict(width=0.5, color='blue'),
                        opacity=0.1,
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add mean path
            mean_path = np.mean(paths, axis=0)
            fig.add_trace(
                go.Scatter(
                    y=mean_path,
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Mean Path'
                ),
                row=1, col=1
            )
        
        # Final value distribution
        if 'final_values' in mc_results:
            final_values = mc_results['final_values']
            
            fig.add_trace(
                go.Histogram(
                    x=final_values,
                    nbinsx=50,
                    name='Final Values',
                    opacity=0.7
                ),
                row=1, col=2
            )
        
        # Convergence analysis
        if 'convergence' in mc_results:
            convergence_data = mc_results['convergence']
            
            fig.add_trace(
                go.Scatter(
                    x=convergence_data['iterations'],
                    y=convergence_data['estimates'],
                    mode='lines',
                    name='Estimate Convergence',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            if 'confidence_bands' in convergence_data:
                upper_band = convergence_data['confidence_bands']['upper']
                lower_band = convergence_data['confidence_bands']['lower']
                
                fig.add_trace(
                    go.Scatter(
                        x=convergence_data['iterations'],
                        y=upper_band,
                        mode='lines',
                        line=dict(color='rgba(0,0,255,0)', width=0),
                        showlegend=False
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=convergence_data['iterations'],
                        y=lower_band,
                        mode='lines',
                        line=dict(color='rgba(0,0,255,0)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(0,0,255,0.2)',
                        name='95% Confidence Band'
                    ),
                    row=2, col=1
                )
        
        # Confidence intervals for different metrics
        if 'confidence_intervals' in mc_results:
            ci_data = mc_results['confidence_intervals']
            metrics = list(ci_data.keys())
            
            for i, metric in enumerate(metrics):
                ci = ci_data[metric]
                
                fig.add_trace(
                    go.Scatter(
                        x=[metric, metric],
                        y=[ci['lower'], ci['upper']],
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=8),
                        name=f'{metric} CI'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            title_text="Monte Carlo Simulation Results",
            showlegend=True
        )
        
        self.figures['monte_carlo'] = fig
        return fig

class RealTimeDashboard:
    """Real-time data dashboard using Streamlit"""
    
    def __init__(self):
        self.data_sources = {}
        
    def setup_streamlit_app(self):
        """Setup Streamlit application"""
        st.set_page_config(
            page_title="Real-Time Finance Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🏦 Real-Time Finance Dashboard")
        
        # Sidebar controls
        st.sidebar.header("Controls")
        
        # Asset selection
        assets = st.sidebar.multiselect(
            "Select Assets",
            ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"],
            default=["AAPL", "GOOGL"]
        )
        
        # Time horizon
        time_horizon = st.sidebar.selectbox(
            "Time Horizon",
            ["1D", "1W", "1M", "3M", "1Y"],
            index=2
        )
        
        # Update frequency
        update_freq = st.sidebar.slider(
            "Update Frequency (seconds)",
            min_value=1,
            max_value=60,
            value=10
        )
        
        return assets, time_horizon, update_freq
        
    def create_real_time_plots(self, market_data: pd.DataFrame):
        """Create real-time market data plots"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Price Chart")
            fig_price = go.Figure()
            
            for column in market_data.columns:
                if 'price' in column.lower():
                    fig_price.add_trace(
                        go.Scatter(
                            x=market_data.index,
                            y=market_data[column],
                            mode='lines',
                            name=column,
                            line=dict(width=2)
                        )
                    )
            
            fig_price.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Price",
                showlegend=True
            )
            
            st.plotly_chart(fig_price, use_container_width=True)
            
        with col2:
            st.subheader("📈 Volume Chart")
            fig_volume = go.Figure()
            
            for column in market_data.columns:
                if 'volume' in column.lower():
                    fig_volume.add_trace(
                        go.Bar(
                            x=market_data.index,
                            y=market_data[column],
                            name=column,
                            opacity=0.7
                        )
                    )
            
            fig_volume.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Volume",
                showlegend=True
            )
            
            st.plotly_chart(fig_volume, use_container_width=True)
            
        # Risk metrics
        st.subheader("⚠️ Risk Metrics")
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            # Calculate simple volatility
            returns = market_data.pct_change().dropna()
            volatility = returns.std().mean() * np.sqrt(252)  # Annualized
            st.metric("Volatility", f"{volatility:.2%}")
            
        with col4:
            # VaR estimation
            var_95 = np.percentile(returns.values.flatten(), 5)
            st.metric("VaR (95%)", f"{var_95:.4f}")
            
        with col5:
            # Sharpe ratio estimation
            sharpe = returns.mean().mean() / returns.std().mean() * np.sqrt(252)
            st.metric("Sharpe Ratio", f"{sharpe:.3f}")
            
        with col6:
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            peak = cumulative.expanding().max()
            drawdown = (cumulative - peak) / peak
            max_dd = drawdown.min().min()
            st.metric("Max Drawdown", f"{max_dd:.2%}")

class RiskDashboard:
    """Specialized risk management dashboard"""
    
    def __init__(self):
        self.risk_alerts = []
        
    def create_risk_monitoring_dashboard(
        self,
        portfolio_data: Dict[str, Any],
        risk_limits: Dict[str, float]
    ) -> Dict[str, go.Figure]:
        """Create comprehensive risk monitoring dashboard"""
        figures = {}
        
        # 1. Portfolio VaR Evolution
        fig_var = go.Figure()
        
        if 'var_history' in portfolio_data:
            var_history = portfolio_data['var_history']
            
            for confidence_level in ['95%', '99%']:
                if confidence_level in var_history:
                    fig_var.add_trace(
                        go.Scatter(
                            x=var_history.index,
                            y=var_history[confidence_level],
                            mode='lines',
                            name=f'VaR {confidence_level}',
                            line=dict(width=2)
                        )
                    )
            
            # Add risk limits
            if 'var_limit' in risk_limits:
                fig_var.add_hline(
                    y=risk_limits['var_limit'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="VaR Limit"
                )
        
        fig_var.update_layout(
            title="Portfolio VaR Evolution",
            xaxis_title="Date",
            yaxis_title="VaR",
            height=400
        )
        figures['var_evolution'] = fig_var
        
        # 2. Risk Attribution
        fig_attribution = go.Figure()
        
        if 'risk_attribution' in portfolio_data:
            attribution = portfolio_data['risk_attribution']
            
            fig_attribution.add_trace(
                go.Bar(
                    x=list(attribution.keys()),
                    y=list(attribution.values()),
                    marker_color='orange',
                    name='Risk Contribution'
                )
            )
        
        fig_attribution.update_layout(
            title="Risk Attribution by Asset",
            xaxis_title="Asset",
            yaxis_title="Risk Contribution",
            height=400
        )
        figures['risk_attribution'] = fig_attribution
        
        # 3. Stress Test Scenarios
        fig_stress = go.Figure()
        
        if 'stress_scenarios' in portfolio_data:
            scenarios = portfolio_data['stress_scenarios']
            
            scenario_names = list(scenarios.keys())
            pnl_values = list(scenarios.values())
            
            colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
            
            fig_stress.add_trace(
                go.Bar(
                    x=scenario_names,
                    y=pnl_values,
                    marker_color=colors,
                    name='Scenario P&L'
                )
            )
            
            # Add stress test limit
            if 'stress_limit' in risk_limits:
                fig_stress.add_hline(
                    y=-risk_limits['stress_limit'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Stress Limit"
                )
        
        fig_stress.update_layout(
            title="Stress Test Scenarios",
            xaxis_title="Scenario",
            yaxis_title="P&L",
            height=400
        )
        figures['stress_tests'] = fig_stress
        
        # 4. Concentration Risk
        fig_concentration = go.Figure()
        
        if 'concentration_metrics' in portfolio_data:
            concentration = portfolio_data['concentration_metrics']
            
            # Herfindahl Index over time
            if 'herfindahl_index' in concentration:
                fig_concentration.add_trace(
                    go.Scatter(
                        x=concentration.index,
                        y=concentration['herfindahl_index'],
                        mode='lines',
                        name='Herfindahl Index',
                        line=dict(color='purple', width=2)
                    )
                )
                
                # Add concentration limit
                if 'concentration_limit' in risk_limits:
                    fig_concentration.add_hline(
                        y=risk_limits['concentration_limit'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Concentration Limit"
                    )
        
        fig_concentration.update_layout(
            title="Concentration Risk Over Time",
            xaxis_title="Date",
            yaxis_title="Herfindahl Index",
            height=400
        )
        figures['concentration_risk'] = fig_concentration
        
        return figures
        
    def generate_risk_alerts(
        self,
        current_metrics: Dict[str, float],
        risk_limits: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Generate risk alerts based on current metrics and limits"""
        alerts = []
        
        for metric, current_value in current_metrics.items():
            limit_key = f"{metric}_limit"
            
            if limit_key in risk_limits:
                limit_value = risk_limits[limit_key]
                
                # Check if limit is breached
                if metric.lower() in ['var', 'expected_shortfall', 'max_drawdown']:
                    # For risk metrics, alert if current value exceeds limit (in absolute terms)
                    if abs(current_value) > abs(limit_value):
                        severity = "HIGH" if abs(current_value) > abs(limit_value) * 1.2 else "MEDIUM"
                        
                        alerts.append({
                            'metric': metric,
                            'current_value': current_value,
                            'limit_value': limit_value,
                            'breach_ratio': abs(current_value) / abs(limit_value),
                            'severity': severity,
                            'message': f"{metric} breach: {current_value:.4f} exceeds limit {limit_value:.4f}"
                        })
                        
                elif metric.lower() in ['concentration', 'leverage']:
                    # For concentration/leverage, alert if above limit
                    if current_value > limit_value:
                        severity = "HIGH" if current_value > limit_value * 1.2 else "MEDIUM"
                        
                        alerts.append({
                            'metric': metric,
                            'current_value': current_value,
                            'limit_value': limit_value,
                            'breach_ratio': current_value / limit_value,
                            'severity': severity,
                            'message': f"{metric} breach: {current_value:.4f} exceeds limit {limit_value:.4f}"
                        })
        
        self.risk_alerts = alerts
        return alerts

# Example usage and testing
if __name__ == "__main__":
    print("Testing Visualization Dashboard...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Sample portfolio data
    initial_value = 1000000
    returns = np.random.normal(0.0008, 0.015, 252)  # Daily returns
    portfolio_values = [initial_value]
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
    portfolio_data = pd.DataFrame({
        'portfolio_value': portfolio_values[1:],
        'allocations': [{'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1}] * 252
    }, index=dates)
    
    print("Created sample portfolio data")
    
    # Test FinanceDashboard
    dashboard = FinanceDashboard("Test Dashboard")
    
    # Create portfolio overview
    portfolio_fig = dashboard.create_portfolio_overview(portfolio_data)
    print("✅ Created portfolio overview figure")
    
    # Sample risk data
    risk_data = {
        'var': {'95%': -25000, '99%': -35000},
        'expected_shortfall': {'95%': -30000, '99%': -40000},
        'risk_contribution': {'AAPL': 0.3, 'GOOGL': 0.25, 'MSFT': 0.2, 'BONDS': 0.15, 'CASH': 0.1},
        'correlation_matrix': pd.DataFrame(
            np.random.rand(5, 5),
            columns=['AAPL', 'GOOGL', 'MSFT', 'BONDS', 'CASH'],
            index=['AAPL', 'GOOGL', 'MSFT', 'BONDS', 'CASH']
        ),
        'stress_tests': {
            '2008 Crisis': -150000,
            'COVID-19': -120000,
            'Tech Crash': -80000,
            'Rate Spike': -60000
        }
    }
    
    # Create risk dashboard
    risk_fig = dashboard.create_risk_metrics_dashboard(risk_data)
    print("✅ Created risk metrics dashboard")
    
    # Sample Monte Carlo results
    mc_results = {
        'paths': np.random.randn(1000, 252).cumsum(axis=1) * 0.01 + 1,
        'final_values': np.random.normal(1.05, 0.15, 10000),
        'convergence': {
            'iterations': list(range(100, 10001, 100)),
            'estimates': np.random.randn(100).cumsum() * 0.001 + 1.05,
            'confidence_bands': {
                'upper': np.random.randn(100).cumsum() * 0.001 + 1.07,
                'lower': np.random.randn(100).cumsum() * 0.001 + 1.03
            }
        },
        'confidence_intervals': {
            'Mean': {'lower': 1.04, 'upper': 1.06},
            'VaR_95': {'lower': 0.82, 'upper': 0.84},
            'VaR_99': {'lower': 0.78, 'upper': 0.80}
        }
    }
    
    # Create Monte Carlo dashboard
    mc_fig = dashboard.create_monte_carlo_results(mc_results)
    print("✅ Created Monte Carlo results dashboard")
    
    # Test RiskDashboard
    risk_dashboard = RiskDashboard()
    
    # Sample portfolio data for risk monitoring
    portfolio_risk_data = {
        'var_history': pd.DataFrame({
            '95%': np.random.normal(-20000, 5000, 252),
            '99%': np.random.normal(-30000, 7500, 252)
        }, index=dates),
        'risk_attribution': {
            'AAPL': 8500, 'GOOGL': 6200, 'MSFT': 5800, 'BONDS': 3500, 'CASH': 1000
        },
        'stress_scenarios': {
            'Market Crash': -180000,
            'Interest Rate Shock': -95000,
            'Currency Crisis': -120000,
            'Liquidity Crisis': -140000
        },
        'concentration_metrics': pd.DataFrame({
            'herfindahl_index': np.random.uniform(0.2, 0.4, 252)
        }, index=dates)
    }
    
    risk_limits = {
        'var_limit': 25000,
        'stress_limit': 150000,
        'concentration_limit': 0.35
    }
    
    # Create risk monitoring dashboard
    risk_figures = risk_dashboard.create_risk_monitoring_dashboard(
        portfolio_risk_data, risk_limits
    )
    print(f"✅ Created {len(risk_figures)} risk monitoring figures")
    
    # Test risk alerts
    current_metrics = {
        'var': -28000,  # Exceeds limit
        'expected_shortfall': -35000,
        'concentration': 0.38,  # Exceeds limit
        'max_drawdown': -0.08
    }
    
    alerts = risk_dashboard.generate_risk_alerts(current_metrics, risk_limits)
    print(f"✅ Generated {len(alerts)} risk alerts")
    
    for alert in alerts:
        print(f"  🚨 {alert['severity']}: {alert['message']}")
    
    print("\nVisualization dashboard test completed!")
