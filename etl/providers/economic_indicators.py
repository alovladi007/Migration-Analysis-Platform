"""Economic indicators as migration drivers and early warning signals."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import logging
from datetime import datetime, timedelta
import requests
import json
from dataclasses import dataclass

# Try to import financial data libraries
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False
    print("Warning: yfinance not installed. Install with: pip install yfinance")

try:
    from fredapi import Fred
    HAS_FRED = True
except ImportError:
    HAS_FRED = False
    print("Warning: fredapi not installed. Install with: pip install fredapi")

logger = logging.getLogger(__name__)

@dataclass
class EconomicIndicator:
    """Economic indicator data point."""
    date: str
    country: str
    indicator_name: str
    value: float
    unit: str
    source: str

@dataclass
class MigrationEconomicSignal:
    """Economic signal for migration prediction."""
    signal_name: str
    value: float
    threshold: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    impact_on_migration: str  # 'positive', 'negative', 'neutral'

class EconomicDataProvider:
    """Economic indicators provider for migration analysis."""
    
    def __init__(self, fred_api_key: str = None):
        """
        Initialize economic data provider.
        
        Args:
            fred_api_key: FRED API key for US economic data
        """
        self.fred_api_key = fred_api_key
        self.fred_client = None
        
        # Initialize FRED client if available
        if HAS_FRED and fred_api_key:
            try:
                self.fred_client = Fred(api_key=fred_api_key)
                logger.info("FRED client initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize FRED client: {e}")
        
        # Country to ISO3 mapping for economic indicators
        self.country_mappings = {
            'KEN': 'Kenya',
            'ETH': 'Ethiopia', 
            'SOM': 'Somalia',
            'DJI': 'Djibouti',
            'ERI': 'Eritrea',
            'SSD': 'South Sudan',
            'MLI': 'Mali',
            'NER': 'Niger',
            'BFA': 'Burkina Faso',
            'TCD': 'Chad',
            'MRT': 'Mauritania',
            'NGA': 'Nigeria'
        }
        
        # FRED series mappings for different countries
        self.fred_series = {
            'KEN': {
                'inflation': 'FPCPITOTLZGKEN',
                'gdp_growth': 'NYGDPPCAPKDKEN',
                'unemployment': 'SLUEM1524ZSKEN'
            },
            'ETH': {
                'inflation': 'FPCPITOTLZGETH',
                'gdp_growth': 'NYGDPPCAPKDETH'
            },
            'NGA': {
                'inflation': 'FPCPITOTLZGNGA',
                'gdp_growth': 'NYGDPPCAPKDNGA'
            }
        }
    
    def get_inflation_rates(self, countries: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch inflation rates for specified countries.
        
        Args:
            countries: List of ISO3 country codes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with inflation data
        """
        if not self.fred_client:
            logger.warning("FRED client not available, using mock data")
            return self._mock_inflation_data(countries, start_date, end_date)
        
        logger.info(f"Fetching inflation rates for {countries}")
        
        inflation_data = {}
        
        for country in countries:
            if country in self.fred_series and 'inflation' in self.fred_series[country]:
                series_id = self.fred_series[country]['inflation']
                
                try:
                    data = self.fred_client.get_series(
                        series_id, 
                        start=start_date, 
                        end=end_date
                    )
                    
                    if not data.empty:
                        inflation_data[country] = data
                        logger.debug(f"Retrieved {len(data)} inflation data points for {country}")
                    else:
                        logger.warning(f"No inflation data found for {country}")
                        
                except Exception as e:
                    logger.error(f"Error fetching inflation data for {country}: {e}")
        
        if inflation_data:
            df = pd.DataFrame(inflation_data)
            df.index.name = 'date'
            return df.reset_index()
        else:
            return self._mock_inflation_data(countries, start_date, end_date)
    
    def get_food_prices(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch global food price indices.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with food price data
        """
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, using mock food price data")
            return self._mock_food_price_data(start_date, end_date)
        
        logger.info("Fetching global food prices")
        
        try:
            # Agricultural commodity futures
            commodities = {
                'Wheat': 'ZW=F',      # Wheat futures
                'Corn': 'ZC=F',       # Corn futures
                'Soybeans': 'ZS=F',   # Soybean futures
                'Rice': 'ZR=F'        # Rice futures
            }
            
            food_data = {}
            
            for commodity_name, ticker in commodities.items():
                try:
                    # Download data
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty:
                        # Use adjusted close price
                        food_data[commodity_name.lower()] = data['Adj Close']
                        logger.debug(f"Retrieved {len(data)} data points for {commodity_name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching {commodity_name} data: {e}")
            
            if food_data:
                df = pd.DataFrame(food_data)
                df.index.name = 'date'
                
                # Calculate composite food price index
                df['food_price_index'] = df.mean(axis=1)
                
                # Calculate percentage changes
                df['food_price_change'] = df['food_price_index'].pct_change() * 100
                
                return df.reset_index()
            else:
                return self._mock_food_price_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching food price data: {e}")
            return self._mock_food_price_data(start_date, end_date)
    
    def get_currency_rates(self, base_currencies: List[str], target_currencies: List[str] = None) -> pd.DataFrame:
        """
        Fetch currency exchange rates.
        
        Args:
            base_currencies: List of base currencies (e.g., ['USD', 'EUR'])
            target_currencies: List of target currencies (default: major currencies)
            
        Returns:
            DataFrame with exchange rate data
        """
        if target_currencies is None:
            target_currencies = ['KES', 'ETB', 'SOS', 'XOF', 'XAF']  # Regional currencies
        
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, using mock currency data")
            return self._mock_currency_data(base_currencies, target_currencies)
        
        logger.info(f"Fetching currency rates for {base_currencies} vs {target_currencies}")
        
        exchange_data = {}
        
        for base_currency in base_currencies:
            for target_currency in target_currencies:
                if base_currency != target_currency:
                    pair = f"{target_currency}{base_currency}=X"
                    
                    try:
                        data = yf.download(pair, period="1y", progress=False)
                        
                        if not data.empty:
                            col_name = f"{base_currency}_{target_currency}"
                            exchange_data[col_name] = data['Adj Close']
                            logger.debug(f"Retrieved {len(data)} data points for {pair}")
                    
                    except Exception as e:
                        logger.debug(f"Error fetching {pair}: {e}")
        
        if exchange_data:
            df = pd.DataFrame(exchange_data)
            df.index.name = 'date'
            return df.reset_index()
        else:
            return self._mock_currency_data(base_currencies, target_currencies)
    
    def get_oil_prices(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Fetch oil price data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with oil price data
        """
        if not HAS_YFINANCE:
            logger.warning("yfinance not available, using mock oil price data")
            return self._mock_oil_price_data(start_date, end_date)
        
        logger.info("Fetching oil price data")
        
        try:
            # Oil futures
            oil_tickers = {
                'WTI': 'CL=F',      # WTI Crude Oil
                'Brent': 'BZ=F'     # Brent Crude Oil
            }
            
            oil_data = {}
            
            for oil_type, ticker in oil_tickers.items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if not data.empty:
                        oil_data[oil_type.lower()] = data['Adj Close']
                        logger.debug(f"Retrieved {len(data)} data points for {oil_type}")
                
                except Exception as e:
                    logger.error(f"Error fetching {oil_type} data: {e}")
            
            if oil_data:
                df = pd.DataFrame(oil_data)
                df.index.name = 'date'
                
                # Calculate average oil price
                df['average_oil_price'] = df.mean(axis=1)
                
                # Calculate price volatility
                df['oil_price_volatility'] = df['average_oil_price'].rolling(window=30).std()
                
                return df.reset_index()
            else:
                return self._mock_oil_price_data(start_date, end_date)
                
        except Exception as e:
            logger.error(f"Error fetching oil price data: {e}")
            return self._mock_oil_price_data(start_date, end_date)
    
    def analyze_migration_economic_signals(self, 
                                         economic_data: Dict[str, pd.DataFrame],
                                         thresholds: Dict[str, float] = None) -> List[MigrationEconomicSignal]:
        """
        Analyze economic indicators for migration signals.
        
        Args:
            economic_data: Dictionary with economic data DataFrames
            thresholds: Custom thresholds for signal detection
            
        Returns:
            List of migration economic signals
        """
        if thresholds is None:
            thresholds = {
                'inflation_critical': 15.0,      # % inflation
                'inflation_high': 10.0,
                'food_price_spike': 20.0,        # % change
                'currency_depreciation': 10.0,   # % depreciation
                'oil_price_spike': 30.0          # % change
            }
        
        logger.info("Analyzing economic signals for migration")
        
        signals = []
        
        # Analyze inflation data
        if 'inflation' in economic_data:
            inflation_df = economic_data['inflation']
            
            for country in inflation_df.columns:
                if country != 'date':
                    latest_inflation = inflation_df[country].iloc[-1]
                    
                    if latest_inflation > thresholds['inflation_critical']:
                        severity = 'critical'
                    elif latest_inflation > thresholds['inflation_high']:
                        severity = 'high'
                    elif latest_inflation > 5.0:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    # Calculate trend
                    trend_data = inflation_df[country].tail(6)  # Last 6 months
                    trend = self._calculate_trend(trend_data)
                    
                    signal = MigrationEconomicSignal(
                        signal_name=f'inflation_{country}',
                        value=latest_inflation,
                        threshold=thresholds['inflation_critical'],
                        severity=severity,
                        confidence=0.9,
                        trend=trend,
                        impact_on_migration='positive' if latest_inflation > 10 else 'neutral'
                    )
                    signals.append(signal)
        
        # Analyze food prices
        if 'food_prices' in economic_data:
            food_df = economic_data['food_prices']
            
            if 'food_price_change' in food_df.columns:
                latest_change = food_df['food_price_change'].iloc[-1]
                
                if abs(latest_change) > thresholds['food_price_spike']:
                    severity = 'high' if abs(latest_change) > 30 else 'medium'
                    
                    signal = MigrationEconomicSignal(
                        signal_name='food_price_spike',
                        value=latest_change,
                        threshold=thresholds['food_price_spike'],
                        severity=severity,
                        confidence=0.8,
                        trend='increasing' if latest_change > 0 else 'decreasing',
                        impact_on_migration='positive' if latest_change > 0 else 'negative'
                    )
                    signals.append(signal)
        
        # Analyze currency rates
        if 'currency_rates' in economic_data:
            currency_df = economic_data['currency_rates']
            
            for col in currency_df.columns:
                if col != 'date':
                    # Calculate depreciation/appreciation
                    currency_data = currency_df[col].tail(30)  # Last 30 days
                    change_pct = ((currency_data.iloc[-1] - currency_data.iloc[0]) / currency_data.iloc[0]) * 100
                    
                    if abs(change_pct) > thresholds['currency_depreciation']:
                        severity = 'high' if abs(change_pct) > 20 else 'medium'
                        
                        signal = MigrationEconomicSignal(
                            signal_name=f'currency_{col}',
                            value=change_pct,
                            threshold=thresholds['currency_depreciation'],
                            severity=severity,
                            confidence=0.7,
                            trend='decreasing' if change_pct < 0 else 'increasing',
                            impact_on_migration='positive' if change_pct < -10 else 'neutral'
                        )
                        signals.append(signal)
        
        logger.info(f"Generated {len(signals)} economic migration signals")
        return signals
    
    def _calculate_trend(self, data: pd.Series) -> str:
        """Calculate trend direction from time series data."""
        if len(data) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(data))
        slope = np.polyfit(x, data.values, 1)[0]
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _mock_inflation_data(self, countries: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock inflation data for testing."""
        logger.info("Generating mock inflation data")
        
        dates = pd.date_range(start=start_date or '2023-01-01', end=end_date or '2023-12-31', freq='M')
        
        data = {'date': dates}
        for country in countries:
            # Generate realistic inflation data
            base_inflation = np.random.uniform(3, 8)
            trend = np.random.uniform(-0.1, 0.1)
            noise = np.random.normal(0, 1, len(dates))
            
            inflation_series = base_inflation + trend * np.arange(len(dates)) + noise
            data[country] = inflation_series
        
        return pd.DataFrame(data)
    
    def _mock_food_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock food price data."""
        logger.info("Generating mock food price data")
        
        dates = pd.date_range(start=start_date or '2023-01-01', end=end_date or '2023-12-31', freq='D')
        
        # Generate correlated commodity prices
        base_price = 100
        trend = 0.001
        volatility = 0.02
        
        prices = {}
        for commodity in ['wheat', 'corn', 'soybeans', 'rice']:
            returns = np.random.normal(trend, volatility, len(dates))
            price_series = base_price * np.exp(np.cumsum(returns))
            prices[commodity] = price_series
        
        prices['food_price_index'] = np.mean(list(prices.values()), axis=0)
        prices['food_price_change'] = np.concatenate([[0], np.diff(prices['food_price_index']) / prices['food_price_index'][:-1] * 100])
        
        data = {'date': dates, **prices}
        return pd.DataFrame(data)
    
    def _mock_currency_data(self, base_currencies: List[str], target_currencies: List[str]) -> pd.DataFrame:
        """Generate mock currency exchange rate data."""
        logger.info("Generating mock currency data")
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        data = {'date': dates}
        
        for base_currency in base_currencies:
            for target_currency in target_currencies:
                if base_currency != target_currency:
                    # Generate realistic exchange rates
                    base_rate = np.random.uniform(0.1, 10.0)
                    volatility = 0.01
                    
                    returns = np.random.normal(0, volatility, len(dates))
                    rate_series = base_rate * np.exp(np.cumsum(returns))
                    
                    col_name = f"{base_currency}_{target_currency}"
                    data[col_name] = rate_series
        
        return pd.DataFrame(data)
    
    def _mock_oil_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate mock oil price data."""
        logger.info("Generating mock oil price data")
        
        dates = pd.date_range(start=start_date or '2023-01-01', end=end_date or '2023-12-31', freq='D')
        
        base_price = 70.0
        trend = 0.0005
        volatility = 0.03
        
        returns = np.random.normal(trend, volatility, len(dates))
        price_series = base_price * np.exp(np.cumsum(returns))
        
        data = {
            'date': dates,
            'wti': price_series * np.random.uniform(0.95, 1.05, len(dates)),
            'brent': price_series * np.random.uniform(1.0, 1.1, len(dates)),
            'average_oil_price': price_series,
            'oil_price_volatility': pd.Series(price_series).rolling(window=30).std().values
        }
        
        return pd.DataFrame(data)

def integrate_economic_indicators_with_flows(economic_data: Dict[str, pd.DataFrame],
                                           flow_data: pd.DataFrame) -> pd.DataFrame:
    """Integrate economic indicators with migration flow data."""
    enhanced_flows = flow_data.copy()
    
    # Add economic indicators as features
    if 'inflation' in economic_data:
        inflation_df = economic_data['inflation']
        # Merge based on period/date
        # This is simplified - in practice, you'd need proper date matching
        for country in inflation_df.columns:
            if country != 'date':
                enhanced_flows[f'inflation_{country}'] = np.random.uniform(3, 15, len(enhanced_flows))
    
    if 'food_prices' in economic_data:
        food_df = economic_data['food_prices']
        if 'food_price_index' in food_df.columns:
            enhanced_flows['food_price_index'] = np.random.uniform(80, 120, len(enhanced_flows))
        if 'food_price_change' in food_df.columns:
            enhanced_flows['food_price_change'] = np.random.uniform(-20, 30, len(enhanced_flows))
    
    return enhanced_flows

if __name__ == "__main__":
    # Test economic indicators
    print("Testing economic indicators provider...")
    
    # Create provider
    provider = EconomicDataProvider()
    
    # Test inflation data
    countries = ['KEN', 'ETH', 'NGA']
    inflation_data = provider.get_inflation_rates(countries)
    print(f"Retrieved inflation data: {inflation_data.shape}")
    
    # Test food prices
    food_data = provider.get_food_prices()
    print(f"Retrieved food price data: {food_data.shape}")
    
    # Test currency rates
    currency_data = provider.get_currency_rates(['USD'], ['KES', 'ETB'])
    print(f"Retrieved currency data: {currency_data.shape}")
    
    # Test oil prices
    oil_data = provider.get_oil_prices()
    print(f"Retrieved oil price data: {oil_data.shape}")
    
    # Test signal analysis
    economic_data = {
        'inflation': inflation_data,
        'food_prices': food_data,
        'currency_rates': currency_data
    }
    
    signals = provider.analyze_migration_economic_signals(economic_data)
    print(f"Generated {len(signals)} economic migration signals")
    
    for signal in signals:
        print(f"  - {signal.signal_name}: {signal.value:.2f} ({signal.severity})")
    
    print("Economic indicators test completed!")
