"""Automated report generation for migration analysis."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from dataclasses import dataclass, asdict

# Try to import visualization libraries
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not installed. Install with: pip install plotly")

try:
    from jinja2 import Template, Environment, FileSystemLoader
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    print("Warning: jinja2 not installed. Install with: pip install jinja2")

try:
    import pdfkit
    HAS_PDFKIT = True
except ImportError:
    HAS_PDFKIT = False
    print("Warning: pdfkit not installed. Install with: pip install pdfkit")

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Report section data."""
    title: str
    content: str
    charts: List[Dict[str, Any]]
    metrics: Dict[str, float]
    recommendations: List[str]

@dataclass
class MigrationReport:
    """Complete migration analysis report."""
    report_id: str
    region: str
    generation_date: str
    period_start: str
    period_end: str
    sections: List[ReportSection]
    summary: Dict[str, Any]
    executive_summary: str
    methodology: str
    limitations: str

class ReportGenerator:
    """Automated report generation for migration analysis."""
    
    def __init__(self, template_dir: str = None):
        """
        Initialize report generator.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir or 'templates'
        self.template_env = None
        
        # Initialize Jinja2 environment if available
        if HAS_JINJA2:
            try:
                self.template_env = Environment(loader=FileSystemLoader(self.template_dir))
                logger.info("Jinja2 template environment initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Jinja2: {e}")
        
        # Report templates
        self.templates = {
            'executive_summary': self._get_executive_summary_template(),
            'methodology': self._get_methodology_template(),
            'limitations': self._get_limitations_template()
        }
    
    def generate_weekly_report(self, 
                             region: str,
                             data: Dict[str, Any],
                             scenario: str = 'baseline',
                             output_format: str = 'html') -> Union[str, bytes]:
        """
        Generate comprehensive weekly migration report.
        
        Args:
            region: Region identifier
            data: Analysis data dictionary
            scenario: Scenario name
            output_format: Output format ('html', 'pdf', 'json')
            
        Returns:
            Generated report content
        """
        logger.info(f"Generating weekly report for {region} ({scenario} scenario)")
        
        # Create report sections
        sections = []
        
        # Executive Summary
        sections.append(self._create_executive_summary_section(data))
        
        # Flow Analysis
        sections.append(self._create_flow_analysis_section(data))
        
        # Uncertainty Analysis
        sections.append(self._create_uncertainty_section(data))
        
        # Risk Assessment
        sections.append(self._create_risk_assessment_section(data))
        
        # Network Analysis
        sections.append(self._create_network_section(data))
        
        # Economic Indicators
        sections.append(self._create_economic_section(data))
        
        # Recommendations
        sections.append(self._create_recommendations_section(data))
        
        # Create complete report
        report = MigrationReport(
            report_id=f"migration_report_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            region=region,
            generation_date=datetime.now().isoformat(),
            period_start=data.get('period_start', '2023-01-01'),
            period_end=data.get('period_end', '2023-12-31'),
            sections=sections,
            summary=self._generate_summary(data),
            executive_summary=self._generate_executive_summary(data, scenario),
            methodology=self.templates['methodology'],
            limitations=self.templates['limitations']
        )
        
        # Generate output
        if output_format == 'html':
            return self._generate_html_report(report)
        elif output_format == 'pdf':
            return self._generate_pdf_report(report)
        elif output_format == 'json':
            return self._generate_json_report(report)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _create_executive_summary_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create executive summary section."""
        predictions = data.get('predictions', {})
        
        # Key metrics
        total_flows = sum(p.get('p50', 0) for p in predictions.values()) if predictions else 0
        high_risk_areas = sum(1 for p in predictions.values() if p.get('p90', 0) > p.get('p50', 0) * 1.5) if predictions else 0
        
        metrics = {
            'total_predicted_flows': total_flows,
            'high_risk_areas': high_risk_areas,
            'uncertainty_level': self._calculate_uncertainty_level(predictions),
            'risk_score': self._calculate_overall_risk_score(data)
        }
        
        content = f"""
        <h2>Executive Summary</h2>
        <p>This report provides a comprehensive analysis of migration patterns and forecasts for the region. 
        Key findings include predicted flows of {total_flows:.0f} individuals across {len(predictions)} administrative areas, 
        with {high_risk_areas} areas identified as high-risk for significant migration events.</p>
        
        <p>The analysis incorporates multiple data sources including historical flows, climate indicators, 
        conflict data, and economic factors to provide robust predictions with uncertainty quantification.</p>
        """
        
        charts = []
        if HAS_PLOTLY and predictions:
            charts.append(self._create_flow_summary_chart(predictions))
        
        recommendations = self._generate_executive_recommendations(metrics)
        
        return ReportSection(
            title="Executive Summary",
            content=content,
            charts=charts,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _create_flow_analysis_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create flow analysis section."""
        predictions = data.get('predictions', {})
        
        content = """
        <h2>Migration Flow Analysis</h2>
        <p>This section analyzes predicted migration flows across administrative boundaries, 
        providing insights into origin-destination patterns and flow magnitudes.</p>
        """
        
        charts = []
        if HAS_PLOTLY and predictions:
            charts.extend([
                self._create_flow_heatmap(predictions),
                self._create_flow_time_series(data.get('historical_flows', pd.DataFrame()))
            ])
        
        # Calculate flow metrics
        flow_metrics = {}
        if predictions:
            flows = [p.get('p50', 0) for p in predictions.values()]
            flow_metrics = {
                'total_flows': sum(flows),
                'average_flow': np.mean(flows),
                'max_flow': max(flows),
                'flow_variance': np.var(flows)
            }
        
        recommendations = self._generate_flow_recommendations(flow_metrics)
        
        return ReportSection(
            title="Migration Flow Analysis",
            content=content,
            charts=charts,
            metrics=flow_metrics,
            recommendations=recommendations
        )
    
    def _create_uncertainty_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create uncertainty analysis section."""
        predictions = data.get('predictions', {})
        
        content = """
        <h2>Uncertainty Analysis</h2>
        <p>This section presents uncertainty quantification for migration predictions, 
        showing prediction intervals (P10-P90) and confidence levels.</p>
        """
        
        charts = []
        if HAS_PLOTLY and predictions:
            charts.extend([
                self._create_uncertainty_bands_chart(predictions),
                self._create_confidence_intervals_chart(predictions)
            ])
        
        # Calculate uncertainty metrics
        uncertainty_metrics = {}
        if predictions:
            uncertainties = []
            for pred in predictions.values():
                p10, p50, p90 = pred.get('p10', 0), pred.get('p50', 0), pred.get('p90', 0)
                if p50 > 0:
                    uncertainty = (p90 - p10) / p50
                    uncertainties.append(uncertainty)
            
            if uncertainties:
                uncertainty_metrics = {
                    'average_uncertainty': np.mean(uncertainties),
                    'max_uncertainty': max(uncertainties),
                    'uncertainty_variance': np.var(uncertainties)
                }
        
        recommendations = self._generate_uncertainty_recommendations(uncertainty_metrics)
        
        return ReportSection(
            title="Uncertainty Analysis",
            content=content,
            charts=charts,
            metrics=uncertainty_metrics,
            recommendations=recommendations
        )
    
    def _create_risk_assessment_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create risk assessment section."""
        content = """
        <h2>Risk Assessment</h2>
        <p>This section evaluates migration risks based on multiple factors including 
        climate stress, conflict levels, and economic indicators.</p>
        """
        
        charts = []
        if HAS_PLOTLY:
            charts.extend([
                self._create_risk_matrix_chart(data.get('risk_factors', {})),
                self._create_risk_trend_chart(data.get('risk_trends', []))
            ])
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(data)
        
        recommendations = self._generate_risk_recommendations(risk_metrics)
        
        return ReportSection(
            title="Risk Assessment",
            content=content,
            charts=charts,
            metrics=risk_metrics,
            recommendations=recommendations
        )
    
    def _create_network_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create network analysis section."""
        network_data = data.get('network_analysis', {})
        
        content = """
        <h2>Migration Network Analysis</h2>
        <p>This section analyzes migration corridors and network connectivity, 
        identifying critical pathways and potential bottlenecks.</p>
        """
        
        charts = []
        if HAS_PLOTLY and network_data:
            charts.extend([
                self._create_network_graph(network_data),
                self._create_corridor_analysis_chart(network_data)
            ])
        
        recommendations = self._generate_network_recommendations(network_data)
        
        return ReportSection(
            title="Migration Network Analysis",
            content=content,
            charts=charts,
            metrics=network_data.get('metrics', {}),
            recommendations=recommendations
        )
    
    def _create_economic_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create economic indicators section."""
        economic_data = data.get('economic_indicators', {})
        
        content = """
        <h2>Economic Indicators</h2>
        <p>This section analyzes economic factors that influence migration decisions, 
        including inflation, food prices, and currency fluctuations.</p>
        """
        
        charts = []
        if HAS_PLOTLY and economic_data:
            charts.extend([
                self._create_economic_trends_chart(economic_data),
                self._create_food_price_chart(economic_data)
            ])
        
        recommendations = self._generate_economic_recommendations(economic_data)
        
        return ReportSection(
            title="Economic Indicators",
            content=content,
            charts=charts,
            metrics=economic_data.get('metrics', {}),
            recommendations=recommendations
        )
    
    def _create_recommendations_section(self, data: Dict[str, Any]) -> ReportSection:
        """Create recommendations section."""
        all_recommendations = []
        
        # Collect recommendations from all sections
        for section_data in [
            self._create_executive_summary_section(data),
            self._create_flow_analysis_section(data),
            self._create_uncertainty_section(data),
            self._create_risk_assessment_section(data),
            self._create_network_section(data),
            self._create_economic_section(data)
        ]:
            all_recommendations.extend(section_data.recommendations)
        
        # Prioritize and deduplicate
        prioritized_recommendations = self._prioritize_recommendations(all_recommendations)
        
        content = f"""
        <h2>Recommendations</h2>
        <p>Based on the comprehensive analysis, the following recommendations are prioritized 
        for immediate action and medium-term planning:</p>
        
        <h3>High Priority</h3>
        <ul>
        """
        
        for rec in prioritized_recommendations.get('high', []):
            content += f"<li>{rec}</li>"
        
        content += """
        </ul>
        
        <h3>Medium Priority</h3>
        <ul>
        """
        
        for rec in prioritized_recommendations.get('medium', []):
            content += f"<li>{rec}</li>"
        
        content += """
        </ul>
        
        <h3>Long-term Planning</h3>
        <ul>
        """
        
        for rec in prioritized_recommendations.get('long_term', []):
            content += f"<li>{rec}</li>"
        
        content += "</ul>"
        
        return ReportSection(
            title="Recommendations",
            content=content,
            charts=[],
            metrics={'total_recommendations': len(all_recommendations)},
            recommendations=[]
        )
    
    # Chart creation methods
    def _create_flow_summary_chart(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Create flow summary chart."""
        if not HAS_PLOTLY:
            return {}
        
        admin_ids = list(predictions.keys())
        p50_values = [predictions[aid].get('p50', 0) for aid in admin_ids]
        
        fig = go.Figure(data=[
            go.Bar(x=admin_ids, y=p50_values, name='Predicted Flows (P50)')
        ])
        
        fig.update_layout(
            title="Predicted Migration Flows by Administrative Area",
            xaxis_title="Administrative Area",
            yaxis_title="Predicted Flow (P50)",
            showlegend=True
        )
        
        return {
            'type': 'plotly',
            'data': fig.to_html(full_html=False, include_plotlyjs=False),
            'title': 'Flow Summary'
        }
    
    def _create_uncertainty_bands_chart(self, predictions: Dict[str, Dict]) -> Dict[str, Any]:
        """Create uncertainty bands chart."""
        if not HAS_PLOTLY:
            return {}
        
        admin_ids = list(predictions.keys())
        p10_values = [predictions[aid].get('p10', 0) for aid in admin_ids]
        p50_values = [predictions[aid].get('p50', 0) for aid in admin_ids]
        p90_values = [predictions[aid].get('p90', 0) for aid in admin_ids]
        
        fig = go.Figure()
        
        # Add uncertainty bands
        fig.add_trace(go.Scatter(
            x=admin_ids + admin_ids[::-1],
            y=p90_values + p10_values[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            name='Uncertainty Band (P10-P90)'
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=admin_ids,
            y=p50_values,
            mode='lines+markers',
            name='Median (P50)',
            line_color='rgb(0,100,80)'
        ))
        
        fig.update_layout(
            title="Migration Flow Predictions with Uncertainty Bands",
            xaxis_title="Administrative Area",
            yaxis_title="Predicted Flow",
            showlegend=True
        )
        
        return {
            'type': 'plotly',
            'data': fig.to_html(full_html=False, include_plotlyjs=False),
            'title': 'Uncertainty Bands'
        }
    
    def _create_risk_matrix_chart(self, risk_factors: Dict[str, float]) -> Dict[str, Any]:
        """Create risk matrix chart."""
        if not HAS_PLOTLY or not risk_factors:
            return {}
        
        factors = list(risk_factors.keys())
        values = list(risk_factors.values())
        
        fig = go.Figure(data=[
            go.Bar(x=factors, y=values, marker_color='red')
        ])
        
        fig.update_layout(
            title="Risk Factor Assessment",
            xaxis_title="Risk Factor",
            yaxis_title="Risk Score",
            showlegend=False
        )
        
        return {
            'type': 'plotly',
            'data': fig.to_html(full_html=False, include_plotlyjs=False),
            'title': 'Risk Matrix'
        }
    
    # Utility methods
    def _calculate_uncertainty_level(self, predictions: Dict[str, Dict]) -> float:
        """Calculate overall uncertainty level."""
        if not predictions:
            return 0.0
        
        uncertainties = []
        for pred in predictions.values():
            p10, p50, p90 = pred.get('p10', 0), pred.get('p50', 0), pred.get('p90', 0)
            if p50 > 0:
                uncertainty = (p90 - p10) / p50
                uncertainties.append(uncertainty)
        
        return np.mean(uncertainties) if uncertainties else 0.0
    
    def _calculate_overall_risk_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall risk score."""
        risk_factors = data.get('risk_factors', {})
        if not risk_factors:
            return 0.0
        
        return np.mean(list(risk_factors.values()))
    
    def _generate_executive_summary(self, data: Dict[str, Any], scenario: str) -> str:
        """Generate executive summary text."""
        predictions = data.get('predictions', {})
        total_flows = sum(p.get('p50', 0) for p in predictions.values()) if predictions else 0
        
        return f"""
        This weekly migration analysis report covers the {data.get('region', 'unknown')} region 
        under the {scenario} scenario. The analysis predicts total migration flows of 
        approximately {total_flows:.0f} individuals across {len(predictions)} administrative areas. 
        
        Key risk factors include climate stress, economic indicators, and conflict levels. 
        The uncertainty analysis shows {self._calculate_uncertainty_level(predictions):.1%} 
        average uncertainty in predictions, indicating {self._get_uncertainty_interpretation(self._calculate_uncertainty_level(predictions))}.
        
        Immediate attention is recommended for areas showing high uncertainty or extreme predictions.
        """
    
    def _get_uncertainty_interpretation(self, uncertainty: float) -> str:
        """Interpret uncertainty level."""
        if uncertainty < 0.3:
            return "relatively confident predictions"
        elif uncertainty < 0.6:
            return "moderate uncertainty in predictions"
        else:
            return "high uncertainty requiring additional monitoring"
    
    def _generate_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate AI-powered recommendations."""
        recommendations = []
        
        predictions = data.get('predictions', {})
        if predictions:
            high_flows = [aid for aid, pred in predictions.items() if pred.get('p50', 0) > 1000]
            if high_flows:
                recommendations.append(f"Prepare emergency response capacity for {len(high_flows)} administrative areas with predicted flows > 1000")
        
        uncertainty = self._calculate_uncertainty_level(predictions)
        if uncertainty > 0.5:
            recommendations.append("Increase monitoring frequency due to high prediction uncertainty")
        
        risk_score = self._calculate_overall_risk_score(data)
        if risk_score > 0.7:
            recommendations.append("Activate high-priority response protocols due to elevated risk factors")
        
        return recommendations
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> Dict[str, List[str]]:
        """Prioritize recommendations by urgency."""
        high_priority_keywords = ['emergency', 'immediate', 'urgent', 'critical', 'activate']
        medium_priority_keywords = ['prepare', 'monitor', 'increase', 'plan']
        
        prioritized = {'high': [], 'medium': [], 'long_term': []}
        
        for rec in recommendations:
            rec_lower = rec.lower()
            if any(keyword in rec_lower for keyword in high_priority_keywords):
                prioritized['high'].append(rec)
            elif any(keyword in rec_lower for keyword in medium_priority_keywords):
                prioritized['medium'].append(rec)
            else:
                prioritized['long_term'].append(rec)
        
        return prioritized
    
    # Template methods
    def _get_executive_summary_template(self) -> str:
        return "Executive summary template content..."
    
    def _get_methodology_template(self) -> str:
        return """
        ## Methodology
        
        This analysis employs a multi-model ensemble approach combining:
        - Gravity models for baseline flow prediction
        - Hawkes processes for shock propagation
        - Quantile regression for uncertainty quantification
        - Network analysis for corridor identification
        - Economic indicators for contextual factors
        
        Data sources include UNHCR refugee statistics, ACLED conflict events, 
        CHIRPS climate data, and economic indicators from various sources.
        """
    
    def _get_limitations_template(self) -> str:
        return """
        ## Limitations
        
        - Predictions are based on historical patterns and may not capture novel events
        - Data quality varies by region and time period
        - Climate and conflict data may have reporting delays
        - Economic indicators are subject to revision
        - Model uncertainty increases with longer prediction horizons
        """
    
    # Output generation methods
    def _generate_html_report(self, report: MigrationReport) -> str:
        """Generate HTML report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Migration Analysis Report - {report.region}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .chart {{ margin: 20px 0; text-align: center; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; text-align: center; }}
                .recommendations {{ background-color: #fff3cd; padding: 20px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Migration Analysis Report</h1>
                <p><strong>Region:</strong> {report.region}</p>
                <p><strong>Generated:</strong> {report.generation_date}</p>
                <p><strong>Period:</strong> {report.period_start} to {report.period_end}</p>
            </div>
            
            <div class="executive-summary">
                <h2>Executive Summary</h2>
                <p>{report.executive_summary}</p>
            </div>
        """
        
        # Add sections
        for section in report.sections:
            html += f"""
            <div class="section">
                <h2>{section.title}</h2>
                <div class="content">
                    {section.content}
                </div>
                
                <div class="charts">
            """
            
            for chart in section.charts:
                if chart.get('type') == 'plotly':
                    html += f'<div class="chart">{chart["data"]}</div>'
            
            html += """
                </div>
                
                <div class="metrics">
            """
            
            for metric_name, metric_value in section.metrics.items():
                html += f'<div class="metric"><strong>{metric_name}:</strong> {metric_value:.2f}</div>'
            
            html += """
                </div>
                
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <ul>
            """
            
            for rec in section.recommendations:
                html += f'<li>{rec}</li>'
            
            html += """
                    </ul>
                </div>
            </div>
            """
        
        # Add methodology and limitations
        html += f"""
            <div class="section">
                <h2>Methodology</h2>
                <p>{report.methodology}</p>
            </div>
            
            <div class="section">
                <h2>Limitations</h2>
                <p>{report.limitations}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_pdf_report(self, report: MigrationReport) -> bytes:
        """Generate PDF report."""
        html_content = self._generate_html_report(report)
        
        if HAS_PDFKIT:
            try:
                pdf_bytes = pdfkit.from_string(html_content, False)
                return pdf_bytes
            except Exception as e:
                logger.error(f"Error generating PDF: {e}")
                return html_content.encode('utf-8')
        else:
            logger.warning("PDF generation not available, returning HTML")
            return html_content.encode('utf-8')
    
    def _generate_json_report(self, report: MigrationReport) -> str:
        """Generate JSON report."""
        # Convert report to dictionary, handling non-serializable objects
        report_dict = asdict(report)
        
        # Convert charts to serializable format
        for section in report_dict['sections']:
            for chart in section['charts']:
                if chart.get('type') == 'plotly':
                    chart['data'] = "Chart data (HTML string)"
        
        return json.dumps(report_dict, indent=2)

# Convenience functions
def generate_weekly_migration_report(region: str,
                                   predictions: Dict[str, Dict],
                                   historical_data: pd.DataFrame = None,
                                   economic_data: Dict[str, Any] = None,
                                   network_data: Dict[str, Any] = None,
                                   output_format: str = 'html') -> Union[str, bytes]:
    """Generate weekly migration report with standard data structure."""
    
    # Prepare data dictionary
    data = {
        'region': region,
        'predictions': predictions,
        'historical_flows': historical_data or pd.DataFrame(),
        'economic_indicators': economic_data or {},
        'network_analysis': network_data or {},
        'risk_factors': {},  # Could be calculated from other data
        'risk_trends': [],
        'period_start': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
        'period_end': datetime.now().strftime('%Y-%m-%d')
    }
    
    # Generate report
    generator = ReportGenerator()
    return generator.generate_weekly_report(
        region=region,
        data=data,
        scenario='baseline',
        output_format=output_format
    )

if __name__ == "__main__":
    # Test report generation
    print("Testing automated report generation...")
    
    # Create sample data
    sample_predictions = {
        'A': {'p10': 50, 'p50': 100, 'p90': 200},
        'B': {'p10': 30, 'p50': 80, 'p90': 150},
        'C': {'p10': 20, 'p50': 60, 'p90': 120}
    }
    
    # Generate report
    report_html = generate_weekly_migration_report(
        region='test_region',
        predictions=sample_predictions,
        output_format='html'
    )
    
    print(f"Generated report with {len(report_html)} characters")
    
    # Save to file for inspection
    with open('test_report.html', 'w') as f:
        f.write(report_html)
    
    print("Report saved to test_report.html")
    print("Automated report generation test completed!")
