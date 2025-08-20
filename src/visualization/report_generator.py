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

# Example usage and testing
if __name__ == "__main__":
    print("Testing Report Generation...")
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Sample portfolio data
    returns = np.random.normal(0.0008, 0.015, 252)
    portfolio_values = [1000000.0]
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
        
    portfolio_data = pd.DataFrame({
        'portfolio_value': portfolio_values[1:],
        'daily_return': returns
    }, index=dates)
    
    print("Generated sample portfolio data")
    
    # Test PDF Report Generator
    print("\nTesting PDF Report Generator:")
    pdf_generator = PDFReportGenerator("Test Financial Report")
    
    # Add executive summary
    executive_summary = {
        'portfolio_value': 1050000,
        'total_return': 0.05,
        'sharpe_ratio': 1.2,
        'max_drawdown': -0.08,
        'var_95': -25000
    }
    
    pdf_generator.add_executive_summary(executive_summary)
    
    # Add portfolio analysis section
    portfolio_analysis_data = {
        'portfolio_data': portfolio_data,
        'drawdown_data': pd.Series(np.random.uniform(-0.1, 0, 252), index=dates),
        'allocation_data': {'Stocks': 60, 'Bonds': 30, 'Cash': 10}
    }
    
    pdf_generator.add_section("Portfolio Analysis", portfolio_analysis_data, 'portfolio_analysis')
    
    # Add risk analysis section
    risk_analysis_data = {
        'var_data': {'95%': -25000, '99%': -35000},
        'risk_attribution': {'AAPL': 8500, 'GOOGL': 6200, 'MSFT': 5800},
        'correlation_matrix': pd.DataFrame(
            np.random.rand(3, 3), 
            columns=['AAPL', 'GOOGL', 'MSFT'],
            index=['AAPL', 'GOOGL', 'MSFT']
        ),
        'stress_tests': {'Market Crash': -150000, 'Rate Spike': -80000}
    }
    
    pdf_generator.add_section("Risk Analysis", risk_analysis_data, 'risk_analysis')
    
    # Add Monte Carlo section
    monte_carlo_data = {
        'paths': np.random.randn(1000, 252).cumsum(axis=1) * 0.01 + 1,
        'final_values': np.random.normal(1.05, 0.15, 10000),
        'convergence': {
            'iterations': list(range(100, 10001, 100)),
            'estimates': np.random.randn(100).cumsum() * 0.001 + 1.05
        },
        'confidence_intervals': {
            'Mean': {'lower': 1.04, 'upper': 1.06},
            'VaR_95': {'lower': 0.82, 'upper': 0.84}
        }
    }
    
    pdf_generator.add_section("Monte Carlo Results", monte_carlo_data, 'monte_carlo')
    
    # Generate PDF report
    try:
        pdf_filename = pdf_generator.generate_report("test_financial_report.pdf")
        print(f"✅ PDF report generated: {pdf_filename}")
    except Exception as e:
        print(f"❌ PDF generation error: {e}")
    
    # Test HTML Report Generator
    print("\nTesting HTML Report Generator:")
    html_generator = HTMLReportGenerator("Test Financial Report - Interactive")
    
    html_generator.add_executive_summary(executive_summary)
    
    # Create a simple Plotly chart for testing
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_data.index,
        y=portfolio_data['portfolio_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(width=2, color='blue')
    ))
    fig.update_layout(
        title='Portfolio Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        height=400
    )
    
    html_generator.add_section("Portfolio Performance Chart", fig, 'chart')
    html_generator.add_section("Portfolio Data Sample", portfolio_data.head(10), 'table')
    
    # Generate HTML report
    try:
        html_filename = html_generator.generate_report("test_financial_report.html")
        print(f"✅ HTML report generated: {html_filename}")
    except Exception as e:
        print(f"❌ HTML generation error: {e}")
    
    print("\nReport generator test completed!")
