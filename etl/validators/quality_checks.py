"""Automated data quality validation and monitoring."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime, timedelta
import json
import warnings
from dataclasses import dataclass, asdict

# Try to import great_expectations
try:
    import great_expectations as ge
    from great_expectations.core import ExpectationSuite
    HAS_GE = True
except ImportError:
    HAS_GE = False
    print("Warning: great_expectations not installed. Install with: pip install great-expectations")

logger = logging.getLogger(__name__)

@dataclass
class QualityCheckResult:
    """Result of a data quality check."""
    check_name: str
    passed: bool
    message: str
    severity: str = 'medium'  # low, medium, high, critical
    details: Optional[Dict] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[QualityCheckResult]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def get_html_report(self) -> str:
        """Generate HTML quality report."""
        severity_colors = {
            'low': '#28a745',
            'medium': '#ffc107', 
            'high': '#fd7e14',
            'critical': '#dc3545'
        }
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Report - {self.dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .check-result {{ margin: 10px 0; padding: 10px; border-radius: 5px; }}
                .passed {{ background-color: #d4edda; border-left: 5px solid #28a745; }}
                .failed {{ background-color: #f8d7da; border-left: 5px solid #dc3545; }}
                .severity-critical {{ border-left-color: #dc3545 !important; }}
                .severity-high {{ border-left-color: #fd7e14 !important; }}
                .severity-medium {{ border-left-color: #ffc107 !important; }}
                .severity-low {{ border-left-color: #28a745 !important; }}
                .details {{ margin-top: 10px; font-size: 0.9em; color: #6c757d; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Data Quality Report</h1>
                <p><strong>Dataset:</strong> {self.dataset_name}</p>
                <p><strong>Generated:</strong> {self.timestamp}</p>
                <p><strong>Overall Status:</strong> {self.passed_checks}/{self.total_checks} checks passed</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <ul>
                    <li>Total Checks: {self.total_checks}</li>
                    <li>Passed: {self.passed_checks}</li>
                    <li>Failed: {self.failed_checks}</li>
                    <li>Success Rate: {(self.passed_checks/self.total_checks*100):.1f}%</li>
                </ul>
            </div>
            
            <h2>Detailed Results</h2>
        """
        
        for result in self.results:
            status_class = 'passed' if result.passed else 'failed'
            severity_class = f'severity-{result.severity}'
            
            html += f"""
            <div class="check-result {status_class} {severity_class}">
                <h3>{result.check_name}</h3>
                <p><strong>Status:</strong> {'✓ PASSED' if result.passed else '✗ FAILED'}</p>
                <p><strong>Severity:</strong> {result.severity.upper()}</p>
                <p><strong>Message:</strong> {result.message}</p>
                <p><strong>Timestamp:</strong> {result.timestamp}</p>
            """
            
            if result.details:
                html += f"""
                <div class="details">
                    <strong>Details:</strong>
                    <pre>{json.dumps(result.details, indent=2)}</pre>
                </div>
                """
            
            html += "</div>"
        
        html += """
        </body>
        </html>
        """
        
        return html

class DataQualityMonitor:
    """Automated data quality validation and monitoring."""
    
    def __init__(self, context_path: str = None):
        """
        Initialize data quality monitor.
        
        Args:
            context_path: Path to Great Expectations context (optional)
        """
        self.context = None
        self.custom_rules = {}
        
        # Initialize Great Expectations if available
        if HAS_GE and context_path:
            try:
                self.context = ge.DataContext(context_path)
                logger.info("Great Expectations context loaded")
            except Exception as e:
                logger.warning(f"Failed to load Great Expectations context: {e}")
        
        # Define built-in validation rules
        self._setup_builtin_rules()
    
    def _setup_builtin_rules(self):
        """Setup built-in data quality rules."""
        self.custom_rules = {
            'flows': {
                'non_negative_flows': {
                    'function': lambda df: (df['flow'] >= 0).all(),
                    'message': 'All flow values must be non-negative',
                    'severity': 'high'
                },
                'temporal_consistency': {
                    'function': lambda df: self._check_temporal_consistency(df),
                    'message': 'Period column should be temporally consistent',
                    'severity': 'medium'
                },
                'completeness': {
                    'function': lambda df: df.isnull().sum().sum() / df.size < 0.05,
                    'message': 'Dataset should have less than 5% missing values',
                    'severity': 'high'
                },
                'flow_range_reasonable': {
                    'function': lambda df: (df['flow'] <= df['flow'].quantile(0.99) * 10).all(),
                    'message': 'Flow values should not exceed 10x the 99th percentile',
                    'severity': 'medium'
                }
            },
            'climate': {
                'spi_range': {
                    'function': lambda df: df['chirps_spi3'].between(-3, 3).all(),
                    'message': 'SPI-3 values should be between -3 and 3',
                    'severity': 'high'
                },
                'temp_anomaly_range': {
                    'function': lambda df: df['era5_tmax_anom'].between(-10, 10).all(),
                    'message': 'Temperature anomalies should be between -10 and 10 degrees',
                    'severity': 'high'
                },
                'climate_completeness': {
                    'function': lambda df: df[['chirps_spi3', 'era5_tmax_anom']].isnull().sum().sum() == 0,
                    'message': 'Climate data should be complete',
                    'severity': 'medium'
                }
            },
            'conflict': {
                'intensity_non_negative': {
                    'function': lambda df: (df['acled_intensity'] >= 0).all(),
                    'message': 'Conflict intensity should be non-negative',
                    'severity': 'high'
                },
                'intensity_reasonable': {
                    'function': lambda df: (df['acled_intensity'] <= 100).all(),
                    'message': 'Conflict intensity should not exceed 100 events per month',
                    'severity': 'medium'
                }
            },
            'population': {
                'population_positive': {
                    'function': lambda df: (df['pop'] > 0).all(),
                    'message': 'Population values should be positive',
                    'severity': 'high'
                },
                'population_reasonable': {
                    'function': lambda df: df['pop'].between(1000, 50000000).all(),
                    'message': 'Population should be between 1,000 and 50 million',
                    'severity': 'medium'
                }
            },
            'access': {
                'access_score_range': {
                    'function': lambda df: df['access_score'].between(0, 1).all(),
                    'message': 'Access scores should be between 0 and 1',
                    'severity': 'high'
                }
            }
        }
    
    def _check_temporal_consistency(self, df: pd.DataFrame) -> bool:
        """Check if periods are temporally consistent."""
        if 'period' not in df.columns:
            return False
        
        try:
            # Convert to datetime and check if sorted
            periods = pd.to_datetime(df['period'], errors='coerce')
            return periods.is_monotonic_increasing
        except:
            return False
    
    def validate_dataset(self, df: pd.DataFrame, 
                        dataset_type: str = 'flows',
                        custom_rules: Dict[str, Any] = None) -> QualityReport:
        """
        Validate dataset against quality rules.
        
        Args:
            df: DataFrame to validate
            dataset_type: Type of dataset ('flows', 'climate', 'conflict', etc.)
            custom_rules: Additional custom rules
            
        Returns:
            QualityReport with validation results
        """
        logger.info(f"Starting quality validation for {dataset_type} dataset with {len(df)} rows")
        
        results = []
        all_rules = self.custom_rules.get(dataset_type, {}).copy()
        
        # Add custom rules if provided
        if custom_rules:
            all_rules.update(custom_rules)
        
        # Run Great Expectations checks if available
        if self.context and HAS_GE:
            ge_results = self._run_ge_validation(df, dataset_type)
            results.extend(ge_results)
        
        # Run built-in and custom rules
        for check_name, rule_config in all_rules.items():
            try:
                if isinstance(rule_config, dict):
                    check_func = rule_config['function']
                    message = rule_config['message']
                    severity = rule_config.get('severity', 'medium')
                else:
                    check_func = rule_config
                    message = f"Custom check: {check_name}"
                    severity = 'medium'
                
                passed = check_func(df)
                
                result = QualityCheckResult(
                    check_name=check_name,
                    passed=passed,
                    message=message,
                    severity=severity,
                    details=self._get_check_details(df, check_name, passed)
                )
                
                results.append(result)
                
                if passed:
                    logger.debug(f"✓ {check_name}: PASSED")
                else:
                    logger.warning(f"✗ {check_name}: FAILED - {message}")
                    
            except Exception as e:
                error_result = QualityCheckResult(
                    check_name=check_name,
                    passed=False,
                    message=f"Check failed with error: {str(e)}",
                    severity='critical',
                    details={'error': str(e)}
                )
                results.append(error_result)
                logger.error(f"✗ {check_name}: ERROR - {e}")
        
        # Generate summary
        total_checks = len(results)
        passed_checks = sum(1 for r in results if r.passed)
        failed_checks = total_checks - passed_checks
        
        summary = {
            'dataset_shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'severity_counts': self._count_by_severity(results)
        }
        
        report = QualityReport(
            dataset_name=dataset_type,
            timestamp=datetime.now().isoformat(),
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            results=results,
            summary=summary
        )
        
        logger.info(f"Quality validation completed: {passed_checks}/{total_checks} checks passed")
        
        return report
    
    def _run_ge_validation(self, df: pd.DataFrame, dataset_type: str) -> List[QualityCheckResult]:
        """Run Great Expectations validation if available."""
        results = []
        
        try:
            # Create expectation suite
            suite_name = f"{dataset_type}_quality_suite"
            suite = self.context.create_expectation_suite(suite_name, overwrite_existing=True)
            
            # Add expectations based on dataset type
            if dataset_type == 'flows':
                suite.add_expectation(ge.expectations.ExpectColumnValuesToNotBeNull('flow'))
                suite.add_expectation(ge.expectations.ExpectColumnValuesToBeBetween('flow', min_value=0))
            
            # Validate
            validator = self.context.get_validator(
                dataframe=df,
                expectation_suite=suite
            )
            validation_result = validator.validate()
            
            # Convert GE results to our format
            for result in validation_result.results:
                ge_result = QualityCheckResult(
                    check_name=result.expectation_config.expectation_type,
                    passed=result.success,
                    message=result.result.get('observed_value', ''),
                    severity='medium',
                    details=result.result
                )
                results.append(ge_result)
                
        except Exception as e:
            logger.warning(f"Great Expectations validation failed: {e}")
        
        return results
    
    def _get_check_details(self, df: pd.DataFrame, check_name: str, passed: bool) -> Dict:
        """Get additional details for a check result."""
        details = {}
        
        if check_name == 'completeness':
            details['missing_counts'] = df.isnull().sum().to_dict()
            details['missing_percentages'] = (df.isnull().sum() / len(df) * 100).to_dict()
        
        elif check_name == 'non_negative_flows':
            details['negative_count'] = (df['flow'] < 0).sum()
            details['min_flow'] = df['flow'].min()
        
        elif check_name == 'flow_range_reasonable':
            details['max_flow'] = df['flow'].max()
            details['q99_flow'] = df['flow'].quantile(0.99)
        
        elif check_name == 'spi_range':
            details['spi_min'] = df['chirps_spi3'].min()
            details['spi_max'] = df['chirps_spi3'].max()
        
        elif check_name == 'temp_anomaly_range':
            details['temp_min'] = df['era5_tmax_anom'].min()
            details['temp_max'] = df['era5_tmax_anom'].max()
        
        return details
    
    def _count_by_severity(self, results: List[QualityCheckResult]) -> Dict[str, int]:
        """Count results by severity level."""
        severity_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for result in results:
            severity_counts[result.severity] += 1
        
        return severity_counts
    
    def add_custom_rule(self, dataset_type: str, rule_name: str, 
                       rule_func: Callable, message: str, severity: str = 'medium'):
        """Add a custom validation rule."""
        if dataset_type not in self.custom_rules:
            self.custom_rules[dataset_type] = {}
        
        self.custom_rules[dataset_type][rule_name] = {
            'function': rule_func,
            'message': message,
            'severity': severity
        }
        
        logger.info(f"Added custom rule '{rule_name}' for dataset type '{dataset_type}'")
    
    def validate_data_pipeline(self, data_sources: Dict[str, pd.DataFrame]) -> Dict[str, QualityReport]:
        """Validate multiple data sources in a pipeline."""
        reports = {}
        
        for source_name, df in data_sources.items():
            logger.info(f"Validating data source: {source_name}")
            
            # Determine dataset type based on source name or columns
            dataset_type = self._infer_dataset_type(source_name, df)
            
            report = self.validate_dataset(df, dataset_type)
            reports[source_name] = report
        
        return reports
    
    def _infer_dataset_type(self, source_name: str, df: pd.DataFrame) -> str:
        """Infer dataset type from source name and columns."""
        source_lower = source_name.lower()
        
        if 'flow' in source_lower or 'od_' in source_lower:
            return 'flows'
        elif 'climate' in source_lower or 'chirps' in source_lower or 'era5' in source_lower:
            return 'climate'
        elif 'conflict' in source_lower or 'acled' in source_lower:
            return 'conflict'
        elif 'pop' in source_lower or 'population' in source_lower:
            return 'population'
        elif 'access' in source_lower:
            return 'access'
        else:
            # Infer from columns
            if 'flow' in df.columns:
                return 'flows'
            elif any(col in df.columns for col in ['chirps_spi3', 'era5_tmax_anom']):
                return 'climate'
            elif 'acled_intensity' in df.columns:
                return 'conflict'
            else:
                return 'general'
    
    def generate_monitoring_dashboard(self, reports: Dict[str, QualityReport]) -> str:
        """Generate monitoring dashboard HTML."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Data Quality Monitoring Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .dashboard { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; }
                .status-good { border-left: 5px solid #28a745; }
                .status-warning { border-left: 5px solid #ffc107; }
                .status-error { border-left: 5px solid #dc3545; }
            </style>
        </head>
        <body>
            <h1>Data Quality Monitoring Dashboard</h1>
            <div class="dashboard">
        """
        
        for source_name, report in reports.items():
            status_class = 'status-good' if report.failed_checks == 0 else 'status-warning' if report.failed_checks < 3 else 'status-error'
            
            html += f"""
            <div class="card {status_class}">
                <h3>{source_name}</h3>
                <p><strong>Status:</strong> {report.passed_checks}/{report.total_checks} checks passed</p>
                <p><strong>Failed:</strong> {report.failed_checks}</p>
                <p><strong>Shape:</strong> {report.summary['dataset_shape']}</p>
                <p><strong>Generated:</strong> {report.timestamp}</p>
            </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html

# Utility functions for data quality
def create_quality_rules_template() -> Dict:
    """Create a template for custom quality rules."""
    return {
        'flows': {
            'custom_rule_example': {
                'function': lambda df: len(df) > 100,
                'message': 'Dataset should have at least 100 rows',
                'severity': 'high'
            }
        }
    }

def validate_migration_data(df: pd.DataFrame, 
                          data_type: str = 'flows',
                          save_report: bool = True,
                          report_path: str = None) -> QualityReport:
    """
    Convenience function for validating migration data.
    
    Args:
        df: DataFrame to validate
        data_type: Type of data ('flows', 'climate', 'conflict', etc.)
        save_report: Whether to save report to file
        report_path: Path to save report (auto-generated if None)
        
    Returns:
        QualityReport object
    """
    monitor = DataQualityMonitor()
    report = monitor.validate_dataset(df, data_type)
    
    if save_report:
        if report_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"quality_report_{data_type}_{timestamp}.html"
        
        with open(report_path, 'w') as f:
            f.write(report.get_html_report())
        
        logger.info(f"Quality report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    # Test data quality validation
    print("Testing data quality validation...")
    
    # Create test data with some quality issues
    test_data = pd.DataFrame({
        'period': ['2020-01', '2020-02', '2020-03', '2020-04'],
        'origin_id': ['A', 'B', 'C', 'D'],
        'dest_id': ['B', 'C', 'D', 'A'],
        'flow': [100, -50, 200, 150],  # Negative value should fail
        'pop_o': [100000, 150000, 200000, 250000],
        'chirps_spi3': [0.5, 1.2, -0.8, np.nan],  # Missing value
        'era5_tmax_anom': [0.3, 0.8, -0.2, 0.1],
        'acled_intensity': [0.1, 0.5, 1.2, 0.3],
        'access_score': [0.7, 0.8, 0.6, 0.9]
    })
    
    # Validate the test data
    report = validate_migration_data(test_data, 'flows', save_report=False)
    
    print(f"Validation completed: {report.passed_checks}/{report.total_checks} checks passed")
    print(f"Failed checks: {report.failed_checks}")
    
    for result in report.results:
        if not result.passed:
            print(f"  ✗ {result.check_name}: {result.message}")
    
    print("Data quality validation test completed!")
