
"""
Comprehensive report generation for financial analysis and Monte Carlo results
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from jinja2 import Template
import base64
import io
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Base report generator class"""
    
    def __init__(self, title: str = "Financial Analysis Report"):
        self.title = title
        self.sections = []
        self.metadata = {
            'created_date': datetime.now(),
            'version': '1.0',
            'author': 'Monte Carlo-Markov Finance System'
        }
        
    def add_section(self, section_title: str, content: Any, section_type: str = 'text'):
        """Add a section to the report"""
        self.sections.append({
            'title': section_title,
            'content': content,
            'type': section_type,
            'timestamp': datetime.now()
        })
        
    def add_executive_summary(self, summary_data: Dict[str, Any]):
        """Add executive summary section"""
        summary_text = self._generate_executive_summary(summary_data)
        self.add_section("Executive Summary", summary_text, "summary")
        
    def _generate_executive_summary(self, data: Dict[str, Any]) -> str:
        """Generate executive summary text from data"""
        summary_parts = []
        
        if 'portfolio_value' in data:
            summary_parts.append(f"Portfolio value: ${data['portfolio_value']:,.2f}")
            
        if 'total_return' in data:
            summary_parts.append(f"Total return: {data['total_return']:.2%}")
            
        if 'sharpe_ratio' in data:
            summary_parts.append(f"Sharpe ratio: {data['sharpe_ratio']:.3f}")
            
        if 'max_drawdown' in data:
            summary_parts.append(f"Maximum drawdown: {data['max_drawdown']:.2%}")
            
        if 'var_95' in data:
            summary_parts.append(f"Value at Risk (95%): ${data['var_95']:,.2f}")
            
        summary = "Key Performance Metrics:\n\n" + "\n".join(f"• {part}" for part in summary_parts)
        
        return summary

class PDFReportGenerator(ReportGenerator):
    """PDF report generator using matplotlib"""
    
    def __init__(self, title: str = "Financial Analysis Report"):
        super().__init__(title)
        plt.style.use('seaborn-v0_8')  # Professional style
        
    def generate_report(self, filename: str) -> str:
        """Generate complete PDF report"""
        logger.info(f"Generating PDF report: {filename}")
        
        with PdfPages(filename) as pdf:
            # Title page
            self._create_title_page(pdf)
            
            # Table of contents
            self._create_table_of_contents(pdf)
            
            # Process all sections
            for section in self.sections:
                if section['type'] == 'summary':
                    self._create_summary_page(pdf, section)
                elif section['type'] == 'chart':
                    self._create_chart_page(pdf, section)
                elif section['type'] == 'table':
                    self._create_table_page(pdf, section)
                elif section['type'] == 'portfolio_analysis':
                    self._create_portfolio_analysis_pages(pdf, section)
                elif section['type'] == 'risk_analysis':
                    self._create_risk_analysis_pages(pdf, section)
                elif section['type'] == 'monte_carlo':
                    self._create_monte_carlo_pages(pdf, section)
                else:
                    self._create_text_page(pdf, section)
                    
            # Appendix
            self._create_appendix(pdf)
            
        logger.info(f"PDF report generated successfully: {filename}")
        return filename
        
    def _create_title_page(self, pdf: PdfPages):
        """Create title page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Title
        ax.text(0.5, 0.8, self.title, fontsize=24, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        
        # Subtitle
        ax.text(0.5, 0.7, 'Comprehensive Financial Analysis Report', 
                fontsize=16, ha='center', va='center', transform=ax.transAxes)
        
        # Date
        ax.text(0.5, 0.6, f"Generated on: {self.metadata['created_date'].strftime('%B %d, %Y')}", 
                fontsize=12, ha='center', va='center', transform=ax.transAxes)
        
        # Author
        ax.text(0.5, 0.5, f"Author: {self.metadata['author']}", 
                fontsize=12, ha='center', va='center', transform=ax.transAxes)
        
        # Version
        ax.text(0.5, 0.4, f"Version: {self.metadata['version']}", 
                fontsize=10, ha='center', va='center', transform=ax.transAxes)
        
        # Logo placeholder
        ax.text(0.5, 0.2, '📊 Monte Carlo-Markov Finance System', 
                fontsize=14, ha='center', va='center', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_table_of_contents(self, pdf: PdfPages):
        """Create table of contents"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.9, 'Table of Contents', fontsize=20, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        
        toc_items = [
            "1. Executive Summary",
            "2. Portfolio Analysis",
            "3. Risk Analysis", 
            "4. Monte Carlo Simulations",
            "5. Performance Metrics",
            "6. Appendix"
        ]
        
        for i, item in enumerate(toc_items):
            y_pos = 0.8 - i * 0.08
            ax.text(0.1, y_pos, item, fontsize=12, 
                    ha='left', va='top', transform=ax.transAxes)
            
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_summary_page(self, pdf: PdfPages, section: Dict):
        """Create executive summary page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Section title
        ax.text(0.5, 0.95, section['title'], fontsize=18, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        
        # Content
        ax.text(0.1, 0.85, section['content'], fontsize=11, 
                ha='left', va='top', transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_portfolio_analysis_pages(self, pdf: PdfPages, section: Dict):
        """Create portfolio analysis pages"""
        data = section['content']
        
        # Page 1: Portfolio Performance
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Portfolio Analysis', fontsize=16, fontweight='bold')
        
        # Portfolio value over time
        if 'portfolio_data' in data:
            portfolio_data = data['portfolio_data']
            ax1.plot(portfolio_data.index, portfolio_data['portfolio_value'], 'b-', linewidth=2)
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Value ($)')
            ax1.grid(True, alpha=0.3)
            
            # Returns distribution
            returns = portfolio_data['portfolio_value'].pct_change().dropna()
            ax2.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_title('Daily Returns Distribution')
            ax2.set_xlabel('Daily Return')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
        # Drawdown analysis
        if 'drawdown_data' in data:
            drawdown_data = data['drawdown_data']
            ax3.fill_between(drawdown_data.index, drawdown_data.values, 0, 
                           alpha=0.5, color='red', label='Drawdown')
            ax3.set_title('Drawdown Analysis')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
            
        # Asset allocation
        if 'allocation_data' in data:
            allocation = data['allocation_data']
            ax4.pie(allocation.values(), labels=allocation.keys(), autopct='%1.1f%%')
            ax4.set_title('Current Asset Allocation')
            
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_risk_analysis_pages(self, pdf: PdfPages, section: Dict):
        """Create risk analysis pages"""
        data = section['content']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Risk Analysis', fontsize=16, fontweight='bold')
        
        # VaR analysis
        if 'var_data' in data:
            var_data = data['var_data']
            confidence_levels = list(var_data.keys())
            var_values = list(var_data.values())
            
            ax1.bar(confidence_levels, var_values, color='red', alpha=0.7)
            ax1.set_title('Value at Risk')
            ax1.set_ylabel('VaR ($)')
            ax1.grid(True, alpha=0.3)
            
        # Risk attribution
        if 'risk_attribution' in data:
            attribution = data['risk_attribution']
            assets = list(attribution.keys())
            contributions = list(attribution.values())
            
            ax2.barh(assets, contributions, color='orange', alpha=0.7)
            ax2.set_title('Risk Attribution')
            ax2.set_xlabel('Risk Contribution')
            ax2.grid(True, alpha=0.3)
            
        # Correlation heatmap
        if 'correlation_matrix' in data:
            corr_matrix = data['correlation_matrix']
            im = ax3.imshow(corr_matrix.values, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
            ax3.set_title('Asset Correlation Matrix')
            ax3.set_xticks(range(len(corr_matrix.columns)))
            ax3.set_yticks(range(len(corr_matrix.index)))
            ax3.set_xticklabels(corr_matrix.columns, rotation=45)
            ax3.set_yticklabels(corr_matrix.index)
            plt.colorbar(im, ax=ax3)
            
        # Stress test results
        if 'stress_tests' in data:
            stress_data = data['stress_tests']
            scenarios = list(stress_data.keys())
            pnl_values = list(stress_data.values())
            
            colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
            ax4.bar(scenarios, pnl_values, color=colors, alpha=0.7)
            ax4.set_title('Stress Test Results')
            ax4.set_ylabel('P&L ($)')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_monte_carlo_pages(self, pdf: PdfPages, section: Dict):
        """Create Monte Carlo simulation pages"""
        data = section['content']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
        
        # Sample paths
        if 'paths' in data:
            paths = data['paths']
            n_display = min(100, paths.shape[0])
            
            for i in range(n_display):
                ax1.plot(paths[i], 'b-', alpha=0.05, linewidth=0.5)
                
            # Mean path
            mean_path = np.mean(paths, axis=0)
            ax1.plot(mean_path, 'r-', linewidth=2, label='Mean Path')
            ax1.set_title('Sample Simulation Paths')
            ax1.set_xlabel('Time Steps')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
        # Final value distribution
        if 'final_values' in data:
            final_values = data['final_values']
            ax2.hist(final_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax2.set_title('Final Value Distribution')
            ax2.set_xlabel('Final Value')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
        # Convergence analysis
        if 'convergence' in data:
            convergence_data = data['convergence']
            ax3.plot(convergence_data['iterations'], convergence_data['estimates'], 'g-', linewidth=2)
            ax3.set_title('Convergence Analysis')
            ax3.set_xlabel('Iterations')
            ax3.set_ylabel('Estimate')
            ax3.grid(True, alpha=0.3)
            
        # Confidence intervals
        if 'confidence_intervals' in data:
            ci_data = data['confidence_intervals']
            metrics = list(ci_data.keys())
            
            for i, metric in enumerate(metrics):
                ci = ci_data[metric]
                ax4.errorbar(i, (ci['upper'] + ci['lower']) / 2, 
                            yerr=(ci['upper'] - ci['lower']) / 2,
                            fmt='o', capsize=5, capthick=2)
                            
            ax4.set_xticks(range(len(metrics)))
            ax4.set_xticklabels(metrics, rotation=45)
            ax4.set_title('Confidence Intervals')
            ax4.set_ylabel('Value')
            ax4.grid(True, alpha=0.3)
            
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_chart_page(self, pdf: PdfPages, section: Dict):
        """Create a page with a chart"""
        fig = section['content']
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_table_page(self, pdf: PdfPages, section: Dict):
        """Create a page with a table"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Section title
        ax.text(0.5, 0.95, section['title'], fontsize=16, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        
        # Table
        table_data = section['content']
        if isinstance(table_data, pd.DataFrame):
            # Convert DataFrame to table
            table = ax.table(cellText=table_data.values,
                           colLabels=table_data.columns,
                           cellLoc='center',
                           loc='center',
                           bbox=[0.1, 0.3, 0.8, 0.5])
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)
            
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_text_page(self, pdf: PdfPages, section: Dict):
        """Create a text page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Section title
        ax.text(0.5, 0.95, section['title'], fontsize=16, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        
        # Content
        ax.text(0.1, 0.85, section['content'], fontsize=11, 
                ha='left', va='top', transform=ax.transAxes, wrap=True)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
    def _create_appendix(self, pdf: PdfPages):
        """Create appendix with methodology and assumptions"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        ax.text(0.5, 0.95, 'Appendix: Methodology and Assumptions', 
                fontsize=16, fontweight='bold',
                ha='center', va='top', transform=ax.transAxes)
        
        appendix_text = """
METHODOLOGY:

1. Monte Carlo Simulations:
   - Number of simulations: 10,000+
   - Random number generation: Mersenne Twister
   - Convergence criteria: Relative tolerance < 1%

2. Risk Calculations:
   - Value at Risk: Historical simulation method
   - Expected Shortfall: Average of tail losses
   - Stress testing: Historical scenario analysis

3. Portfolio Optimization:
   - Mean-variance optimization
   - Risk parity constraints
   - Transaction cost consideration

ASSUMPTIONS:

1. Market Data:
   - Normal distribution of returns (where applicable)
   - Constant volatility (adjusted for regime changes)
   - Liquid markets assumption

2. Risk Models:
   - Correlation stability over time horizon
   - No model risk consideration
   - Linear exposure assumptions

DISCLAIMER:

This report is for informational purposes only and does not constitute 
investment advice. Past performance does not guarantee future results.
All investments carry risk of loss.
        """
        
        ax.text(0.1, 0.85, appendix_text, fontsize=9, 
                ha='left', va='top', transform=ax.transAxes)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

class HTMLReportGenerator(ReportGenerator):
    """HTML report generator with interactive charts"""
    
    def __init__(self, title: str = "Financial Analysis Report"):
        super().__init__(title)
        self.template = self._get_html_template()
        
    def generate_report(self, filename: str) -> str:
        """Generate complete HTML report"""
        logger.info(f"Generating HTML report: {filename}")
        
        # Convert sections to HTML
        html_sections = []
        
        for section in self.sections:
            if section['type'] == 'chart':
                html_content = self._plotly_to_html(section['content'])
            elif section['type'] == 'table':
                html_content = self._dataframe_to_html(section['content'])
            else:
                html_content = f"<div class='text-section'><p>{section['content']}</p></div>"
                
            html_sections.append({
                'title': section['title'],
                'content': html_content,
                'type': section['type']
            })
            
        # Render template
        html_content = self.template.render(
            title=self.title,
            sections=html_sections,
            metadata=self.metadata,
            generated_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"HTML report generated successfully: {filename}")
        return filename
        
    def _plotly_to_html(self, fig) -> str:
        """Convert Plotly figure to HTML div"""
        return pio.to_html(fig, include_plotlyjs='cdn', div_id=f"plot_{id(fig)}")
        
    def _dataframe_to_html(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to HTML table"""
        return df.to_html(classes='table table-striped table-hover', escape=False)
        
    def _get_html_template(self) -> Template:
        """Get HTML template for report"""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { font-family: Arial, sans-serif; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; }
        .section { margin: 2rem 0; padding: 1rem; border-left: 4px solid #007bff; }
        .chart-container { margin: 1rem 0; }
        .table { margin: 1rem 0; }
        .metadata { background: #f8f9fa; padding: 1rem; border-radius: 0.25rem; }
        .toc { background: #fff; padding: 1rem; border: 1px solid #dee2e6; border-radius: 0.25rem; }
        .footer { background: #343a40; color: white; padding: 1rem; margin-top: 3rem; }
    </style>
</head>
<body>
    <div class="header text-center">
        <h1>{{ title }}</h1>
        <p class="lead">Comprehensive Financial Analysis Report</p>
        <p>Generated on {{ generated_date }}</p>
    </div>
    
    <div class="container-fluid">
        <div class="row">
            <div class="col-md-3">
                <div class="toc sticky-top" style="top: 20px;">
                    <h5>Table of Contents</h5>
                    <ul class="list-unstyled">
                        {% for section in sections %}
                        <li><a href="#section-{{ loop.index }}" class="text-decoration-none">{{ section.title }}</a></li>
                        {% endfor %}
                    </ul>
                </div>
            </div>
            
            <div class="col-md-9">
                {% for section in sections %}
                <div class="section" id="section-{{ loop.index }}">
                    <h2>{{ section.title }}</h2>
                    <div class="content">
                        {{ section.content | safe }}
                    </div>
                </div>
                {% endfor %}
                
                <div class="metadata">
                    <h5>Report Metadata</h5>
                    <ul>
                        <li><strong>Author:</strong> {{ metadata.author }}</li>
                        <li><strong>Version:</strong> {{ metadata.version }}</li>
                        <li><strong>Created:</strong> {{ metadata.created_date.strftime('%Y-%m-%d %H:%M:%S') }}</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    
    <div class="footer text-center">
        <p>&copy; 2025 Monte Carlo-Markov Finance System. All rights reserved.</p>
        <p><small>This report is for informational purposes only and does not constitute investment advice.</small></p>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """
        return Template(template_str)

### **src/visualization/plotting_utils.py**
```
"""
Plotting utilities for financial data visualization
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class PlottingUtils:
    """Utility functions for financial plotting"""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        self.color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
    def setup_financial_plot_style(self):
        """Setup professional financial plot style"""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False
        })
        
    def format_currency_axis(self, ax, axis: str = 'y'):
        """Format axis to display currency values"""
        if axis == 'y':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
    def format_percentage_axis(self, ax, axis: str = 'y'):
        """Format axis to display percentage values"""
        if axis == 'y':
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        else:
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
    def format_date_axis(self, ax, axis: str = 'x'):
        """Format axis to display dates nicely"""
        if axis == 'x':
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

class StaticPlots:
    """Static plotting functions using matplotlib"""
    
    def __init__(self):
        self.utils = PlottingUtils()
        self.utils.setup_financial_plot_style()
        
    def plot_price_series(
        self,
        data: pd.DataFrame,
        price_column: str = 'close',
        volume_column: Optional[str] = None,
        title: str = "Price Series"
    ) -> plt.Figure:
        """Plot price series with optional volume"""
        if volume_column:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                         gridspec_kw={'height_ratios': })
        else:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
        # Price plot
        ax1.plot(data.index, data[price_column], linewidth=2, color='blue')
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        self.utils.format_currency_axis(ax1)
        self.utils.format_date_axis(ax1)
        
        # Volume plot (if provided)
        if volume_column and volume_column in data.columns:
            ax2.bar(data.index, data[volume_column], alpha=0.7, color='gray')
            ax2.set_ylabel('Volume', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            self.utils.format_date_axis(ax2)
            
        plt.tight_layout()
        return fig
        
    def plot_returns_analysis(
        self,
        returns: pd.Series,
        title: str = "Returns Analysis"
    ) -> plt.Figure:
        """Plot comprehensive returns analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Returns time series
        ax1.plot(returns.index, returns, linewidth=1, alpha=0.7, color='blue')
        ax1.set_title('Returns Over Time')
        ax1.set_ylabel('Daily Return')
        self.utils.format_percentage_axis(ax1)
        self.utils.format_date_axis(ax1)
        
        # Returns distribution
        ax2.hist(returns.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        self.utils.format_percentage_axis(ax2, 'x')
        
        # QQ plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normal Distribution)')
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252)
        ax4.plot(rolling_vol.index, rolling_vol, linewidth=2, color='red')
        ax4.set_title('30-Day Rolling Volatility (Annualized)')
        ax4.set_ylabel('Volatility')
        self.utils.format_percentage_axis(ax4)
        self.utils.format_date_axis(ax4)
        
        plt.tight_layout()
        return fig
        
    def plot_portfolio_performance(
        self,
        portfolio_data: pd.DataFrame,
        benchmark_data: Optional[pd.DataFrame] = None
    ) -> plt.Figure:
        """Plot portfolio performance analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Portfolio Performance Analysis', fontsize=16, fontweight='bold')
        
        # Portfolio value
        ax1.plot(portfolio_data.index, portfolio_data['portfolio_value'], 
                linewidth=2, color='blue', label='Portfolio')
        
        if benchmark_data is not None:
            ax1.plot(benchmark_data.index, benchmark_data['value'], 
                    linewidth=2, color='red', linestyle='--', label='Benchmark')
            ax1.legend()
            
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Value')
        self.utils.format_currency_axis(ax1)
        self.utils.format_date_axis(ax1)
        
        # Drawdown
        peak = portfolio_data['portfolio_value'].expanding().max()
        drawdown = (portfolio_data['portfolio_value'] - peak) / peak
        
        ax2.fill_between(portfolio_data.index, drawdown, 0, alpha=0.5, color='red')
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown')
        self.utils.format_percentage_axis(ax2)
        self.utils.format_date_axis(ax2)
        
        # Monthly returns
        monthly_returns = portfolio_data['portfolio_value'].resample('M').last().pct_change().dropna()
        colors = ['green' if ret > 0 else 'red' for ret in monthly_returns]
        
        ax3.bar(monthly_returns.index, monthly_returns, color=colors, alpha=0.7)
        ax3.set_title('Monthly Returns')
        ax3.set_ylabel('Monthly Return')
        self.utils.format_percentage_axis(ax3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Risk-return scatter (if benchmark provided)
        if benchmark_data is not None:
            portfolio_returns = portfolio_data['portfolio_value'].pct_change().dropna()
            benchmark_returns = benchmark_data['value'].pct_change().dropna()
            
            port_vol = portfolio_returns.std() * np.sqrt(252)
            port_ret = portfolio_returns.mean() * 252
            bench_vol = benchmark_returns.std() * np.sqrt(252)
            bench_ret = benchmark_returns.mean() * 252
            
            ax4.scatter(port_vol, port_ret, s=100, color='blue', label='Portfolio')
            ax4.scatter(bench_vol, bench_ret, s=100, color='red', label='Benchmark')
            ax4.set_title('Risk-Return Profile')
            ax4.set_xlabel('Volatility (Annualized)')
            ax4.set_ylabel('Return (Annualized)')
            self.utils.format_percentage_axis(ax4, 'both')
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No benchmark data available', 
                    ha='center', va='center', transform=ax4.transAxes)
            
        plt.tight_layout()
        return fig
        
    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        title: str = "Asset Correlation Matrix"
    ) -> plt.Figure:
        """Plot correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

class InteractivePlots:
    """Interactive plotting functions using Plotly"""
    
    def __init__(self):
        self.color_sequence = px.colors.qualitative.Set1
        
    def plot_interactive_price_series(
        self,
        data: pd.DataFrame,
        price_columns: List[str],
        volume_column: Optional[str] = None,
        title: str = "Interactive Price Series"
    ) -> go.Figure:
        """Create interactive price series plot"""
        if volume_column:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price', 'Volume'),
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
            
        # Add price series
        for i, column in enumerate(price_columns):
            if column in data.columns:
                if volume_column:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[column],
                            mode='lines',
                            name=column,
                            line=dict(width=2, color=self.color_sequence[i % len(self.color_sequence)])
                        ),
                        row=1, col=1
                    )
                else:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data[column],
                            mode='lines',
                            name=column,
                            line=dict(width=2, color=self.color_sequence[i % len(self.color_sequence)])
                        )
                    )
                    
        # Add volume if provided
        if volume_column and volume_column in data.columns:
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data[volume_column],
                    name='Volume',
                    opacity=0.7,
                    marker_color='gray'
                ),
                row=2, col=1
            )
            
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Price",
            hovermode='x unified',
            height=600 if volume_column else 400
        )
        
        return fig
        
    def plot_interactive_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "Returns Distribution Analysis"
    ) -> go.Figure:
        """Create interactive returns distribution plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Returns Over Time', 'Distribution', 'Box Plot', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Returns time series
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=returns,
                mode='lines',
                name='Returns',
                line=dict(width=1, color='blue'),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Distribution histogram
        fig.add_trace(
            go.Histogram(
                x=returns.dropna(),
                nbinsx=50,
                name='Distribution',
                opacity=0.7,
                marker_color='blue'
            ),
            row=1, col=2
        )
        
        # Box plot
        fig.add_trace(
            go.Box(
                y=returns.dropna(),
                name='Returns',
                boxpoints='outliers',
                marker_color='blue'
            ),
            row=2, col=1
        )
        
        # Cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        fig.add_trace(
            go.Scatter(
                x=returns.index,
                y=cumulative_returns,
                mode='lines',
                name='Cumulative Returns',
                line=dict(width=2, color='green')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
        
    def plot_interactive_risk_dashboard(
        self,
        risk_data: Dict[str, Any],
        title: str = "Interactive Risk Dashboard"
    ) -> go.Figure:
        """Create interactive risk dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('VaR Evolution', 'Risk Attribution', 'Stress Tests', 'Correlation Heatmap'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "heatmap"}]]
        )
        
        # VaR evolution
        if 'var_history' in risk_data:
            var_history = risk_data['var_history']
            for confidence_level in var_history.columns:
                fig.add_trace(
                    go.Scatter(
                        x=var_history.index,
                        y=var_history[confidence_level],
                        mode='lines',
                        name=f'VaR {confidence_level}',
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
                
        # Risk attribution
        if 'risk_attribution' in risk_data:
            attribution = risk_data['risk_attribution']
            fig.add_trace(
                go.Bar(
                    x=list(attribution.keys()),
                    y=list(attribution.values()),
                    name='Risk Contribution',
                    marker_color='orange'
                ),
                row=1, col=2
            )
            
        # Stress tests
        if 'stress_tests' in risk_data:
            stress_data = risk_data['stress_tests']
            colors = ['red' if val < 0 else 'green' for val in stress_data.values()]
            
            fig.add_trace(
                go.Bar(
                    x=list(stress_data.keys()),
                    y=list(stress_data.values()),
                    name='Stress Test P&L',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
        # Correlation heatmap
        if 'correlation_matrix' in risk_data:
            corr_matrix = risk_data['correlation_matrix']
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    name='Correlations'
                ),
                row=2, col=2
            )
            
        fig.update_layout(
            title=title,
            height=800,
            showlegend=True
        )
        
        return fig

# Example usage and testing
if __name__ == "__main__":
    print("Testing Plotting Utilities...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Sample price data
    returns = np.random.normal(0.0008, 0.015, 252)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    # Sample volume data
    volumes = np.random.randint(1000000, 5000000, 252)
    
    market_data = pd.DataFrame({
        'close': prices[1:],
        'volume': volumes,
        'returns': returns
    }, index=dates)
    
    print("Generated sample market data")
    
    # Test Static Plots
    static_plots = StaticPlots()
    
    # Test price series plot
    print("Testing static price series plot...")
    price_fig = static_plots.plot_price_series(
        market_data, 'close', 'volume', 'Sample Price Series'
    )
    print("✅ Static price series plot created")
    
    # Test returns analysis
    print("Testing returns analysis plot...")
    returns_fig = static_plots.plot_returns_analysis(
        market_data['returns'], 'Returns Analysis'
    )
    print("✅ Returns analysis plot created")
    
    # Test portfolio performance
    print("Testing portfolio performance plot...")
    portfolio_data = pd.DataFrame({
        'portfolio_value': np.cumprod(1 + market_data['returns']) * 1000000
    }, index=dates)
    
    portfolio_fig = static_plots.plot_portfolio_performance(portfolio_data)
    print("✅ Portfolio performance plot created")
    
    # Test correlation heatmap
    print("Testing correlation heatmap...")
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    correlation_matrix = pd.DataFrame(
        np.random.rand(5, 5),
        columns=assets,
        index=assets
    )
    # Make it symmetric
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
    np.fill_diagonal(correlation_matrix.values, 1)
    
    corr_fig = static_plots.plot_correlation_heatmap(correlation_matrix)
    print("✅ Correlation heatmap created")
    
    # Test Interactive Plots
    interactive_plots = InteractivePlots()
    
    # Test interactive price series
    print("Testing interactive price series...")
    interactive_price_fig = interactive_plots.plot_interactive_price_series(
        market_data, ['close'], 'volume', 'Interactive Price Series'
    )
    print("✅ Interactive price series created")
    
    # Test interactive returns distribution
    print("Testing interactive returns distribution...")
    interactive_returns_fig = interactive_plots.plot_interactive_returns_distribution(
        market_data['returns'], 'Interactive Returns Analysis'
    )
    print("✅ Interactive returns distribution created")
    
    # Test interactive risk dashboard
    print("Testing interactive risk dashboard...")
    risk_data = {
        'var_history': pd.DataFrame({
            '95%': np.random.normal(-20000, 5000, 252),
            '99%': np.random.normal(-30000, 7500, 252)
        }, index=dates),
        'risk_attribution': {
            'AAPL': 8500, 'GOOGL': 6200, 'MSFT': 5800, 'BONDS': 3500, 'CASH': 1000
        },
        'stress_tests': {
            'Market Crash': -180000,
            'Interest Rate Shock': -95000,
            'Currency Crisis': -120000
        },
        'correlation_matrix': correlation_matrix
    }
    
    interactive_risk_fig = interactive_plots.plot_interactive_risk_dashboard(
        risk_data, 'Interactive Risk Dashboard'
    )
    print("✅ Interactive risk dashboard created")
    
    # Test Report Generators
    print("\nTesting Report Generators...")
    
    # PDF Report
    pdf_generator = PDFReportGenerator("Test Financial Report")
    
    # Add sections
    executive_summary = {
        'portfolio_value': 1050000,
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'var_95': -25000
    }
    
    pdf_generator.add_executive_summary(executive_summary)
    
    # Add portfolio analysis
    portfolio_analysis_data = {
        'portfolio_data': portfolio_data,
        'drawdown_data': pd.Series(
            np.random.uniform(-0.1, 0, 252), index=dates
        ),
        'allocation_data': {'Stocks': 60, 'Bonds': 30, 'Cash': 10}
    }
    
    pdf_generator.add_section(
        "Portfolio Analysis", 
        portfolio_analysis_data, 
        'portfolio_analysis'
    )
    
    # Add risk analysis
    risk_analysis_data = {
        'var_data': {'95%': -25000, '99%': -35000},
        'risk_attribution': risk_data['risk_attribution'],
        'correlation_matrix': correlation_matrix,
        'stress_tests': risk_data['stress_tests']
    }
    
    pdf_generator.add_section(
        "Risk Analysis",
        risk_analysis_data,
        'risk_analysis'
    )
    
    # Generate PDF report
    pdf_filename = "test_financial_report.pdf"
    try:
        pdf_generator.generate_report(pdf_filename)
        print(f"✅ PDF report generated: {pdf_filename}")
    except Exception as e:
        print(f"❌ PDF report generation failed: {e}")
    
    # HTML Report
    html_generator = HTMLReportGenerator("Test Financial Report - HTML")
    
    html_generator.add_executive_summary(executive_summary)
    html_generator.add_section("Interactive Price Chart", interactive_price_fig, 'chart')
    html_generator.add_section("Risk Dashboard", interactive_risk_fig, 'chart')
    html_generator.add_section("Portfolio Data", portfolio_data.head(10), 'table')
    
    # Generate HTML report
    html_filename = "test_financial_report.html"
    try:
        html_generator.generate_report(html_filename)
        print(f"✅ HTML report generated: {html_filename}")
    except Exception as e:
        print(f"❌ HTML report generation failed: {e}")
    
    print("\nVisualization components test completed!")
```

## **16. Complete Test Suite Implementation**

### **tests/__init__.py**
```
"""
Comprehensive test suite for Monte Carlo-Markov Finance System
"""

# Test configuration
import os
import sys
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Test fixtures and utilities
from .conftest import *
```

### **tests/conftest.py**
```
"""
Pytest configuration and fixtures for the test suite
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Test data fixtures
@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate realistic price series
    returns = np.random.normal(0.0008, 0.015, 252)
    prices = [100.0]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    volumes = np.random.randint(1000000, 5000000, 252)
    
    return pd.DataFrame({
        'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices[1:]],
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'close': prices[1:],
        'volume': volumes
    }, index=dates)

@pytest.fixture
def sample_returns_data():
    """Generate sample returns data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    returns = np.random.normal(0.0008, 0.015, 252)
    return pd.Series(returns, index=dates)

@pytest.fixture
def sample_portfolio_data():
    """Generate sample portfolio data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    returns = np.random.normal(0.0008, 0.015, 252)
    portfolio_values = [1000000.0]
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
    return pd.DataFrame({
        'portfolio_value': portfolio_values[1:],
        'daily_return': returns,
        'allocation': [{'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1}] * 252
    }, index=dates)

@pytest.fixture
def sample_correlation_matrix():
    """Generate sample correlation matrix"""
    np.random.seed(42)
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    # Generate random correlation matrix
    random_matrix = np.random.randn(5, 5)
    correlation_matrix = np.corrcoef(random_matrix)
    
    return pd.DataFrame(correlation_matrix, columns=assets, index=assets)

@pytest.fixture
def sample_monte_carlo_results():
    """Generate sample Monte Carlo results"""
    np.random.seed(42)
    n_paths = 1000
    n_steps = 252
    
    # Generate sample paths
    returns = np.random.normal(0.0008, 0.015, (n_paths, n_steps))
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = 100.0
    
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * (1 + returns[:, t])
        
    return {
        'paths': paths,
        'final_values': paths[:, -1],
        'returns': returns,
        'statistics': {
            'mean': np.mean(paths[:, -1]),
            'std': np.std(paths[:, -1]),
            'var_95': np.percentile(paths[:, -1], 5),
            'var_99': np.percentile(paths[:, -1], 1)
        }
    }

@pytest.fixture
def sample_backtest_data():
    """Generate sample backtest data"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate market data
    returns = np.random.normal(0.0008, 0.015, 252)
    prices = [100.0]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
        
    market_data = pd.DataFrame({
        'close': prices[1:],
        'open': [p * (1 + np.random.normal(0, 0.002)) for p in prices[1:]],
        'high': [p * (1 + abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'low': [p * (1 - abs(np.random.normal(0, 0.008))) for p in prices[1:]],
        'volume': np.random.randint(1000000, 5000000, 252)
    }, index=dates)
    
    # Generate portfolio data
    portfolio_returns = np.random.normal(0.001, 0.018, 252)
    portfolio_values = [100000.0]
    for ret in portfolio_returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
    portfolio_data = pd.DataFrame({
        'portfolio_value': portfolio_values[1:],
        'daily_return': portfolio_returns
    }, index=dates)
    
    return {
        'market_data': market_data,
        'portfolio_data': portfolio_data
    }

# Test configuration
def pytest_configure(config):
    """Configure pytest"""
    # Set random seed for reproducible tests
    np.random.seed(42)
    
    # Configure warnings
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
```

### **tests/test_monte_carlo_engine.py**
```
"""
Tests for Monte Carlo simulation engines
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from monte_carlo_engine.base_monte_carlo import BaseMonteCarloEngine
from monte_carlo_engine.gbm_engine import GeometricBrownianMotionEngine
from monte_carlo_engine.path_dependent import PathDependentEngine
from monte_carlo_engine.multi_asset import MultiAssetEngine

class TestBaseMonteCarloEngine:
    """Test base Monte Carlo engine functionality"""
    
    def test_initialization(self):
        """Test engine initialization"""
        engine = BaseMonteCarloEngine(n_simulations=1000, n_steps=252)
        
        assert engine.n_simulations == 1000
        assert engine.n_steps == 252
        assert engine.random_seed is None
        assert not engine.antithetic_variates
        
    def test_set_random_seed(self):
        """Test random seed setting"""
        engine = BaseMonteCarloEngine(n_simulations=1000, n_steps=252)
        engine.set_random_seed(42)
        
        assert engine.random_seed == 42
        
        # Test reproducibility
        result1 = engine.generate_random_numbers((100, 10))
        engine.set_random_seed(42)  # Reset seed
        result2 = engine.generate_random_numbers((100, 10))
        
        np.testing.assert_array_equal(result1, result2)
        
    def test_generate_random_numbers(self):
        """Test random number generation"""
        engine = BaseMonteCarloEngine(n_simulations=1000, n_steps=252)
        engine.set_random_seed(42)
        
        # Test normal distribution
        normal_nums = engine.generate_random_numbers((1000, 252), distribution='normal')
        assert normal_nums.shape == (1000, 252)
        assert abs(np.mean(normal_nums)) < 0.1  # Should be close to 0
        assert abs(np.std(normal_nums) - 1) < 0.1  # Should be close to 1
        
        # Test uniform distribution
        uniform_nums = engine.generate_random_numbers((1000, 252), distribution='uniform')
        assert uniform_nums.shape == (1000, 252)
        assert np.all(uniform_nums >= 0) and np.all(uniform_nums <= 1)
        
    def test_antithetic_variates(self):
        """Test antithetic variates variance reduction"""
        engine = BaseMonteCarloEngine(n_simulations=1000, n_steps=252, antithetic_variates=True)
        engine.set_random_seed(42)
        
        random_nums = engine.generate_random_numbers((1000, 252))
        
        # With antithetic variates, second half should be negative of first half
        first_half = random_nums[:500]
        second_half = random_nums[500:]
        
        np.testing.assert_array_almost_equal(first_half, -second_half, decimal=10)

class TestGeometricBrownianMotionEngine:
    """Test GBM engine functionality"""
    
    def test_initialization(self):
        """Test GBM engine initialization"""
        engine = GeometricBrownianMotionEngine(
            n_simulations=1000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        
        assert engine.initial_price == 100.0
        assert engine.drift == 0.05
        assert engine.volatility == 0.2
        
    def test_simulate_paths(self, sample_monte_carlo_results):
        """Test path simulation"""
        engine = GeometricBrownianMotionEngine(
            n_simulations=1000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        engine.set_random_seed(42)
        
        paths = engine.simulate_paths()
        
        assert paths.shape == (1000, 253)  # n_steps + 1
        assert np.all(paths[:, 0] == 100.0)  # Initial price
        assert np.all(paths > 0)  # All prices should be positive
        
    def test_theoretical_moments(self):
        """Test theoretical moments calculation"""
        engine = GeometricBrownianMotionEngine(
            n_simulations=10000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        
        # Calculate theoretical final value moments
        T = 1.0  # 1 year
        expected_final = 100.0 * np.exp(0.05 * T)
        expected_variance = 100.0**2 * np.exp(2 * 0.05 * T) * (np.exp(0.2**2 * T) - 1)
        
        # Simulate and compare
        engine.set_random_seed(42)
        paths = engine.simulate_paths()
        final_values = paths[:, -1]
        
        empirical_mean = np.mean(final_values)
        empirical_variance = np.var(final_values)
        
        # Should be close (within 5% for large simulation)
        assert abs(empirical_mean - expected_final) / expected_final < 0.05
        assert abs(empirical_variance - expected_variance) / expected_variance < 0.1

class TestPathDependentEngine:
    """Test path-dependent options engine"""
    
    def test_asian_option_pricing(self, sample_monte_carlo_results):
        """Test Asian option pricing"""
        engine = PathDependentEngine(
            n_simulations=10000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        engine.set_random_seed(42)
        
        # Price Asian call option
        result = engine.price_asian_option(
            strike=100.0,
            option_type='call',
            averaging_start=0,
            risk_free_rate=0.03,
            time_to_maturity=1.0
        )
        
        assert 'price' in result
        assert 'std_error' in result
        assert 'confidence_interval' in result
        assert result['price'] > 0  # Option should have positive value
        assert result['std_error'] > 0  # Should have some uncertainty
        
    def test_barrier_option_pricing(self):
        """Test barrier option pricing"""
        engine = PathDependentEngine(
            n_simulations=10000,
            n_steps=252,
            initial_price=100.0,
            drift=0.05,
            volatility=0.2
        )
        engine.set_random_seed(42)
        
        # Price up-and-out call
        result = engine.price_barrier_option(
            strike=100.0,
            barrier=120.0,
            option_type='call',
            barrier_type='up-and-out',
            risk_free_rate=0.03,
            time_to_maturity=1.0
        )
        
        assert 'price' in result
        assert result['price'] >= 0  # Barrier option value should be non-negative
        
        # Up-and-out should be cheaper than vanilla call
        vanilla_result = engine.price_european_option(
            strike=100.0,
            option_type='call',
            risk_free_rate=0.03,
            time_to_maturity=1.0
        )
        
        assert result['price'] <= vanilla_result['price']

class TestMultiAssetEngine:
    """Test multi-asset Monte Carlo engine"""
    
    def test_initialization(self, sample_correlation_matrix):
        """Test multi-asset engine initialization"""
        initial_prices = [100.0, 200.0, 150.0]
        drifts = [0.05, 0.06, 0.04]
        volatilities = [0.2, 0.25, 0.18]
        
        engine = MultiAssetEngine(
            n_simulations=1000,
            n_steps=252,
            initial_prices=initial_prices,
            drifts=drifts,
            volatilities=volatilities,
            correlation_matrix=sample_correlation_matrix.iloc[:3, :3].values
        )
        
        assert len(engine.initial_prices) == 3
        assert len(engine.drifts) == 3
        assert len(engine.volatilities) == 3
        assert engine.correlation_matrix.shape == (3, 3)
        
    def test_correlated_path_simulation(self, sample_correlation_matrix):
        """Test correlated path simulation"""
        initial_prices = [100.0, 200.0]
        drifts = [0.05, 0.06]
        volatilities = [0.2, 0.25]
        correlation_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        engine = MultiAssetEngine(
            n_simulations=10000,
            n_steps=252,
            initial_prices=initial_prices,
            drifts=drifts,
            volatilities=volatilities,
            correlation_matrix=correlation_matrix
        )
        engine.set_random_seed(42)
        
        paths = engine.simulate_correlated_paths()
        
        assert paths.shape == (10000, 253, 2)  # (sims, steps+1, assets)
        
        # Check initial prices
        np.testing.assert_array_equal(paths[:, 0, :], [100.0, 200.0])
        
        # Check correlation of returns
        returns1 = np.diff(np.log(paths[:, :, 0]), axis=1)
        returns2 = np.diff(np.log(paths[:, :, 1]), axis=1)
        
        empirical_corr = np.corrcoef(returns1.flatten(), returns2.flatten())
        
        # Should be close to target correlation (within tolerance for MC)
        assert abs(empirical_corr - 0.5) < 0.05
        
    def test_basket_option_pricing(self, sample_correlation_matrix):
        """Test basket option pricing"""
        initial_prices = [100.0, 100.0, 100.0]
        drifts = [0.05, 0.05, 0.05]
        volatilities = [0.2, 0.2, 0.2]
        weights = [1/3, 1/3, 1/3]
        
        engine = MultiAssetEngine(
            n_simulations=10000,
            n_steps=252,
            initial_prices=initial_prices,
            drifts=drifts,
            volatilities=volatilities,
            correlation_matrix=np.eye(3)  # Independent assets
        )
        engine.set_random_seed(42)
        
        result = engine.price_basket_option(
            weights=weights,
            strike=100.0,
            option_type='call',
            risk_free_rate=0.03,
            time_to_maturity=1.0
        )
        
        assert 'price' in result
        assert result['price'] > 0
        assert 'greeks' in result

if __name__ == "__main__":
    pytest.main([__file__])
```
