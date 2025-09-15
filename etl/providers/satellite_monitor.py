"""Satellite-based displacement detection and monitoring."""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

# Try to import Google Earth Engine
try:
    import ee
    HAS_EE = True
except ImportError:
    HAS_EE = False
    print("Warning: earthengine-api not installed. Install with: pip install earthengine-api")

# Try to import raster processing libraries
try:
    import rasterio
    import rasterio.features
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not installed. Install with: pip install rasterio")

logger = logging.getLogger(__name__)

@dataclass
class SatelliteObservation:
    """Satellite observation result."""
    timestamp: str
    location: Tuple[float, float]
    observation_type: str
    value: float
    confidence: float
    metadata: Dict

@dataclass
class DisplacementDetection:
    """Displacement detection result."""
    area_id: str
    detection_date: str
    displacement_type: str  # 'new_settlement', 'camp_expansion', 'depopulation'
    magnitude: float
    confidence: float
    coordinates: List[Tuple[float, float]]
    affected_area_km2: float

class DisplacementMonitor:
    """Satellite-based displacement monitoring using Earth Engine."""
    
    def __init__(self, service_account_path: str = None):
        """
        Initialize displacement monitor.
        
        Args:
            service_account_path: Path to Google Earth Engine service account JSON
        """
        self.initialized = False
        
        if HAS_EE:
            try:
                if service_account_path:
                    # Initialize with service account
                    credentials = ee.ServiceAccountCredentials(None, service_account_path)
                    ee.Initialize(credentials)
                else:
                    # Initialize with user credentials
                    ee.Initialize()
                
                self.initialized = True
                logger.info("Google Earth Engine initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Earth Engine: {e}")
        else:
            logger.warning("Google Earth Engine not available - using mock data")
    
    def detect_settlement_changes(self, 
                                bbox: Tuple[float, float, float, float],
                                date_start: str,
                                date_end: str,
                                threshold: float = 0.2) -> List[DisplacementDetection]:
        """
        Detect new settlements or camp expansions using Sentinel-2 imagery.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            date_start: Start date (YYYY-MM-DD)
            date_end: End date (YYYY-MM-DD)
            threshold: Threshold for change detection (0-1)
            
        Returns:
            List of displacement detections
        """
        if not self.initialized:
            return self._mock_settlement_detection(bbox, date_start, date_end)
        
        logger.info(f"Detecting settlement changes in bbox {bbox} from {date_start} to {date_end}")
        
        try:
            # Create geometry from bounding box
            geometry = ee.Geometry.Rectangle(bbox)
            
            # Get Sentinel-2 imagery
            collection = (ee.ImageCollection('COPERNICUS/S2_SR')
                         .filterBounds(geometry)
                         .filterDate(date_start, date_end)
                         .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
            
            if collection.size().getInfo() == 0:
                logger.warning("No suitable imagery found for the date range")
                return []
            
            # Calculate built-up area index
            def calculate_bui(image):
                """Calculate Built-up Index."""
                ndbi = image.normalizedDifference(['B11', 'B8'])  # NDBI
                ndvi = image.normalizedDifference(['B8', 'B4'])   # NDVI
                bui = ndbi.subtract(ndvi)
                return bui.rename('BUI').copyProperties(image, ['system:time_start'])
            
            # Apply BUI calculation
            bui_collection = collection.map(calculate_bui)
            
            # Get first and last images
            first_image = bui_collection.sort('system:time_start').first()
            last_image = bui_collection.sort('system:time_start', False).first()
            
            # Calculate change
            change_image = last_image.subtract(first_image)
            
            # Threshold for new settlements
            new_settlements = change_image.gt(threshold)
            
            # Extract polygons of change
            detections = self._extract_change_polygons(new_settlements, geometry)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in settlement change detection: {e}")
            return self._mock_settlement_detection(bbox, date_start, date_end)
    
    def monitor_agricultural_stress(self, 
                                  region_geom: Union[Dict, List],
                                  date_start: str,
                                  date_end: str) -> Dict:
        """
        Monitor agricultural stress using MODIS NDVI.
        
        Args:
            region_geom: Region geometry (GeoJSON or coordinates)
            date_start: Start date (YYYY-MM-DD)
            date_end: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary with vegetation stress metrics
        """
        if not self.initialized:
            return self._mock_agricultural_stress(region_geom, date_start, date_end)
        
        logger.info(f"Monitoring agricultural stress from {date_start} to {date_end}")
        
        try:
            # Convert geometry
            if isinstance(region_geom, dict):
                geometry = ee.Geometry(region_geom)
            else:
                geometry = ee.Geometry.Polygon(region_geom)
            
            # Get MODIS NDVI data
            modis = ee.ImageCollection('MODIS/006/MOD13Q1')
            
            # Filter by date and geometry
            ndvi_collection = (modis
                             .filterDate(date_start, date_end)
                             .filterBounds(geometry)
                             .select('NDVI'))
            
            # Calculate vegetation condition index
            def calculate_vci(image):
                """Calculate Vegetation Condition Index."""
                ndvi = image.select('NDVI').multiply(0.0001)
                return ndvi.rename('VCI').copyProperties(image, ['system:time_start'])
            
            vci_collection = ndvi_collection.map(calculate_vci)
            
            # Calculate statistics
            mean_vci = vci_collection.mean()
            
            # Reduce to region
            stats = mean_vci.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            # Calculate anomalies
            historical_mean = self._get_historical_ndvi_mean(geometry, date_start, date_end)
            
            result = stats.getInfo()
            result['historical_mean'] = historical_mean
            result['anomaly'] = result.get('VCI', 0) - historical_mean
            
            return result
            
        except Exception as e:
            logger.error(f"Error in agricultural stress monitoring: {e}")
            return self._mock_agricultural_stress(region_geom, date_start, date_end)
    
    def detect_displacement_camps(self, 
                                bbox: Tuple[float, float, float, float],
                                date_start: str,
                                date_end: str) -> List[DisplacementDetection]:
        """
        Detect displacement camps using satellite imagery.
        
        Args:
            bbox: Bounding box
            date_start: Start date
            date_end: End date
            
        Returns:
            List of camp detections
        """
        if not self.initialized:
            return self._mock_camp_detection(bbox, date_start, date_end)
        
        logger.info(f"Detecting displacement camps in bbox {bbox}")
        
        try:
            geometry = ee.Geometry.Rectangle(bbox)
            
            # Use Landsat 8 for higher resolution
            landsat = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
                      .filterBounds(geometry)
                      .filterDate(date_start, date_end)
                      .filter(ee.Filter.lt('CLOUD_COVER', 20)))
            
            if landsat.size().getInfo() == 0:
                logger.warning("No suitable Landsat imagery found")
                return []
            
            # Calculate indices for camp detection
            def calculate_indices(image):
                """Calculate spectral indices for camp detection."""
                ndvi = image.normalizedDifference(['B5', 'B4'])
                ndbi = image.normalizedDifference(['B6', 'B5'])
                ndwi = image.normalizedDifference(['B3', 'B5'])
                
                # Camp detection index (high NDBI, low NDVI, moderate NDWI)
                camp_index = ndbi.subtract(ndvi.multiply(0.5)).add(ndwi.multiply(0.3))
                
                return image.addBands([
                    ndvi.rename('NDVI'),
                    ndbi.rename('NDBI'),
                    ndwi.rename('NDWI'),
                    camp_index.rename('CampIndex')
                ])
            
            # Apply indices
            indexed_collection = landsat.map(calculate_indices)
            
            # Get median composite
            median_image = indexed_collection.median()
            
            # Threshold for camp detection
            camp_threshold = 0.3
            potential_camps = median_image.select('CampIndex').gt(camp_threshold)
            
            # Extract camp polygons
            camps = self._extract_change_polygons(potential_camps, geometry)
            
            return camps
            
        except Exception as e:
            logger.error(f"Error in camp detection: {e}")
            return self._mock_camp_detection(bbox, date_start, date_end)
    
    def monitor_water_resources(self, 
                              region_geom: Union[Dict, List],
                              date_start: str,
                              date_end: str) -> Dict:
        """
        Monitor water resources using satellite data.
        
        Args:
            region_geom: Region geometry
            date_start: Start date
            date_end: End date
            
        Returns:
            Water resource metrics
        """
        if not self.initialized:
            return self._mock_water_monitoring(region_geom, date_start, date_end)
        
        logger.info(f"Monitoring water resources from {date_start} to {date_end}")
        
        try:
            if isinstance(region_geom, dict):
                geometry = ee.Geometry(region_geom)
            else:
                geometry = ee.Geometry.Polygon(region_geom)
            
            # Use MODIS for water monitoring
            modis = ee.ImageCollection('MODIS/006/MOD09A1')
            
            # Filter and select relevant bands
            water_collection = (modis
                              .filterDate(date_start, date_end)
                              .filterBounds(geometry)
                              .select(['sur_refl_b02', 'sur_refl_b04', 'sur_refl_b07']))
            
            def calculate_water_indices(image):
                """Calculate water-related indices."""
                # Normalized Difference Water Index (NDWI)
                ndwi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b04'])
                
                # Modified NDWI
                mndwi = image.normalizedDifference(['sur_refl_b02', 'sur_refl_b07'])
                
                return image.addBands([
                    ndwi.rename('NDWI'),
                    mndwi.rename('MNDWI')
                ])
            
            # Apply indices
            indexed_collection = water_collection.map(calculate_water_indices)
            
            # Calculate mean water coverage
            mean_ndwi = indexed_collection.select('NDWI').mean()
            mean_mndwi = indexed_collection.select('MNDWI').mean()
            
            # Water threshold
            water_threshold = 0.2
            
            # Calculate water area
            water_mask = mean_ndwi.gt(water_threshold)
            
            # Reduce to region
            water_stats = water_mask.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=500,
                maxPixels=1e9
            )
            
            # Calculate total area
            total_stats = ee.Image.pixelArea().reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=geometry,
                scale=500,
                maxPixels=1e9
            )
            
            water_area = water_stats.getInfo()
            total_area = total_stats.getInfo()
            
            water_percentage = (water_area.get('NDWI', 0) / total_area.get('area', 1)) * 100
            
            return {
                'water_percentage': water_percentage,
                'mean_ndwi': mean_ndwi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=500
                ).getInfo(),
                'mean_mndwi': mean_mndwi.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=500
                ).getInfo()
            }
            
        except Exception as e:
            logger.error(f"Error in water monitoring: {e}")
            return self._mock_water_monitoring(region_geom, date_start, date_end)
    
    def _extract_change_polygons(self, change_image: ee.Image, geometry: ee.Geometry) -> List[DisplacementDetection]:
        """Extract polygons from change detection image."""
        try:
            # Convert to vectors
            vectors = change_image.reduceToVectors(
                geometry=geometry,
                scale=30,
                geometryType='polygon',
                eightConnected=False,
                labelProperty='change'
            )
            
            # Get feature collection
            features = vectors.getInfo()
            
            detections = []
            for i, feature in enumerate(features.get('features', [])):
                coords = feature['geometry']['coordinates'][0]  # First ring
                
                # Calculate area
                area_km2 = self._calculate_polygon_area(coords)
                
                detection = DisplacementDetection(
                    area_id=f"detection_{i}",
                    detection_date=datetime.now().isoformat(),
                    displacement_type='new_settlement',
                    magnitude=1.0,  # Placeholder
                    confidence=0.7,  # Placeholder
                    coordinates=coords,
                    affected_area_km2=area_km2
                )
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error extracting change polygons: {e}")
            return []
    
    def _calculate_polygon_area(self, coordinates: List[List[float]]) -> float:
        """Calculate polygon area in km²."""
        try:
            # Simple approximation using shoelace formula
            x_coords = [coord[0] for coord in coordinates]
            y_coords = [coord[1] for coord in coordinates]
            
            area = 0.5 * abs(sum(x_coords[i] * y_coords[i+1] - x_coords[i+1] * y_coords[i] 
                                for i in range(-1, len(x_coords)-1)))
            
            # Convert from degrees² to km² (approximate)
            area_km2 = area * (111.32 ** 2)  # 1 degree ≈ 111.32 km
            
            return area_km2
            
        except Exception as e:
            logger.error(f"Error calculating polygon area: {e}")
            return 0.0
    
    def _get_historical_ndvi_mean(self, geometry: ee.Geometry, date_start: str, date_end: str) -> float:
        """Get historical NDVI mean for anomaly calculation."""
        try:
            # Get 10-year historical average
            historical_start = (datetime.strptime(date_start, '%Y-%m-%d') - timedelta(days=365*10)).strftime('%Y-%m-%d')
            historical_end = date_start
            
            modis = ee.ImageCollection('MODIS/006/MOD13Q1')
            historical_collection = (modis
                                   .filterDate(historical_start, historical_end)
                                   .filterBounds(geometry)
                                   .select('NDVI'))
            
            historical_mean = historical_collection.mean()
            
            stats = historical_mean.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=250,
                maxPixels=1e9
            )
            
            return stats.getInfo().get('NDVI', 0) * 0.0001  # Scale factor
            
        except Exception as e:
            logger.error(f"Error getting historical NDVI mean: {e}")
            return 0.5  # Default value
    
    # Mock methods for testing without Earth Engine
    def _mock_settlement_detection(self, bbox: Tuple, date_start: str, date_end: str) -> List[DisplacementDetection]:
        """Mock settlement detection for testing."""
        logger.info("Using mock settlement detection")
        
        detections = []
        for i in range(2):  # Mock 2 detections
            detection = DisplacementDetection(
                area_id=f"mock_settlement_{i}",
                detection_date=date_end,
                displacement_type='new_settlement',
                magnitude=0.5 + i * 0.3,
                confidence=0.8 - i * 0.1,
                coordinates=[(bbox[0] + i * 0.01, bbox[1] + i * 0.01)],
                affected_area_km2=1.5 + i * 0.5
            )
            detections.append(detection)
        
        return detections
    
    def _mock_agricultural_stress(self, region_geom: Union[Dict, List], date_start: str, date_end: str) -> Dict:
        """Mock agricultural stress monitoring."""
        logger.info("Using mock agricultural stress data")
        
        return {
            'VCI': 0.4 + np.random.random() * 0.3,
            'historical_mean': 0.5,
            'anomaly': -0.1 + np.random.random() * 0.2,
            'stress_level': 'moderate'
        }
    
    def _mock_camp_detection(self, bbox: Tuple, date_start: str, date_end: str) -> List[DisplacementDetection]:
        """Mock camp detection."""
        logger.info("Using mock camp detection")
        
        detections = []
        for i in range(1):  # Mock 1 camp
            detection = DisplacementDetection(
                area_id=f"mock_camp_{i}",
                detection_date=date_end,
                displacement_type='camp_expansion',
                magnitude=0.8,
                confidence=0.9,
                coordinates=[(bbox[0] + 0.005, bbox[1] + 0.005)],
                affected_area_km2=2.0
            )
            detections.append(detection)
        
        return detections
    
    def _mock_water_monitoring(self, region_geom: Union[Dict, List], date_start: str, date_end: str) -> Dict:
        """Mock water monitoring."""
        logger.info("Using mock water monitoring data")
        
        return {
            'water_percentage': 15.0 + np.random.random() * 10.0,
            'mean_ndwi': {'NDWI': 0.2 + np.random.random() * 0.1},
            'mean_mndwi': {'MNDWI': 0.15 + np.random.random() * 0.1},
            'water_stress_level': 'moderate'
        }

def create_displacement_alert(detection: DisplacementDetection) -> Dict:
    """Create displacement alert from detection."""
    return {
        'alert_type': 'displacement_detected',
        'timestamp': detection.detection_date,
        'location': detection.coordinates[0] if detection.coordinates else None,
        'displacement_type': detection.displacement_type,
        'magnitude': detection.magnitude,
        'confidence': detection.confidence,
        'affected_area_km2': detection.affected_area_km2,
        'severity': 'high' if detection.magnitude > 0.7 else 'medium' if detection.magnitude > 0.4 else 'low'
    }

def integrate_satellite_with_flows(displacement_detections: List[DisplacementDetection],
                                 flow_data: pd.DataFrame) -> pd.DataFrame:
    """Integrate satellite displacement detections with flow data."""
    enhanced_flows = flow_data.copy()
    
    # Add displacement indicators
    enhanced_flows['satellite_displacement_detected'] = False
    enhanced_flows['displacement_magnitude'] = 0.0
    enhanced_flows['displacement_confidence'] = 0.0
    
    for detection in displacement_detections:
        # Match with flow data based on location (simplified)
        if detection.coordinates:
            lat, lon = detection.coordinates[0]
            
            # This is a simplified matching - in practice, you'd use spatial joins
            # For now, randomly assign to some flows
            mask = enhanced_flows.sample(frac=0.1).index  # 10% of flows
            enhanced_flows.loc[mask, 'satellite_displacement_detected'] = True
            enhanced_flows.loc[mask, 'displacement_magnitude'] = detection.magnitude
            enhanced_flows.loc[mask, 'displacement_confidence'] = detection.confidence
    
    return enhanced_flows

if __name__ == "__main__":
    # Test satellite monitoring
    print("Testing satellite displacement monitoring...")
    
    # Create monitor
    monitor = DisplacementMonitor()
    
    # Test settlement detection
    bbox = (36.0, 4.0, 37.0, 5.0)  # Example bbox in Kenya
    detections = monitor.detect_settlement_changes(
        bbox=bbox,
        date_start='2023-01-01',
        date_end='2023-12-31'
    )
    
    print(f"Detected {len(detections)} settlement changes")
    for detection in detections:
        print(f"  - {detection.displacement_type}: {detection.affected_area_km2:.2f} km²")
    
    # Test agricultural stress monitoring
    region_geom = {'type': 'Polygon', 'coordinates': [[[36.0, 4.0], [37.0, 4.0], [37.0, 5.0], [36.0, 5.0], [36.0, 4.0]]]}
    stress_data = monitor.monitor_agricultural_stress(
        region_geom=region_geom,
        date_start='2023-01-01',
        date_end='2023-12-31'
    )
    
    print(f"Agricultural stress data: {stress_data}")
    
    # Test water monitoring
    water_data = monitor.monitor_water_resources(
        region_geom=region_geom,
        date_start='2023-01-01',
        date_end='2023-12-31'
    )
    
    print(f"Water monitoring data: {water_data}")
    
    print("Satellite monitoring test completed!")
