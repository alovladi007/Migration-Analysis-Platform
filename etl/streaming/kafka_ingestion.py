"""Real-time event processing for immediate alerts."""
from kafka import KafkaConsumer, KafkaProducer
import json
from datetime import datetime
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MobilityStreamProcessor:
    def __init__(self, bootstrap_servers: list = None):
        """Initialize Kafka stream processor for real-time mobility monitoring."""
        self.bootstrap_servers = bootstrap_servers or ['localhost:9092']
        
        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            'mobility-events',
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='mobility-monitor'
        )
        
        # Initialize Kafka producer for alerts
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        # Alert thresholds
        self.thresholds = {
            'sudden_movement': 1000,  # people/day
            'conflict_intensity': 5.0,
            'climate_shock': 2.0,    # SPI deviation
            'temperature_spike': 3.0,  # degrees above normal
            'border_crossing_spike': 500  # people/hour
        }
        
        # Historical baseline for anomaly detection
        self.baseline_window = 24  # hours
        self.recent_events = []
        
    def process_stream(self):
        """Main processing loop for real-time events."""
        logger.info("Starting mobility stream processing...")
        
        try:
            for message in self.consumer:
                event = message.value
                event['processed_at'] = datetime.now().isoformat()
                
                # Store recent events for baseline comparison
                self.recent_events.append(event)
                if len(self.recent_events) > self.baseline_window * 10:  # Keep last 240 events
                    self.recent_events.pop(0)
                
                # Check for anomalies
                anomalies = self.detect_anomalies(event)
                
                if anomalies:
                    alert = self.create_alert(event, anomalies)
                    self.send_alert(alert)
                    
                # Process different event types
                self.process_event_by_type(event)
                
        except KeyboardInterrupt:
            logger.info("Stream processing interrupted")
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self.cleanup()
    
    def detect_anomalies(self, event: Dict[str, Any]) -> list:
        """Detect anomalies in real-time events."""
        anomalies = []
        
        if event['type'] == 'displacement':
            # Check for sudden movement spikes
            if event.get('count', 0) > self.thresholds['sudden_movement']:
                anomalies.append({
                    'type': 'sudden_movement',
                    'severity': 'high',
                    'value': event['count'],
                    'threshold': self.thresholds['sudden_movement']
                })
            
            # Check against historical baseline
            if self.recent_events:
                baseline = self.calculate_baseline('displacement')
                if event.get('count', 0) > baseline * 3:
                    anomalies.append({
                        'type': 'baseline_deviation',
                        'severity': 'medium',
                        'value': event['count'],
                        'baseline': baseline,
                        'deviation': event['count'] / baseline
                    })
        
        elif event['type'] == 'conflict':
            if event.get('intensity', 0) > self.thresholds['conflict_intensity']:
                anomalies.append({
                    'type': 'conflict_spike',
                    'severity': 'high',
                    'value': event['intensity'],
                    'threshold': self.thresholds['conflict_intensity']
                })
        
        elif event['type'] == 'climate':
            if abs(event.get('spi_anomaly', 0)) > self.thresholds['climate_shock']:
                anomalies.append({
                    'type': 'climate_shock',
                    'severity': 'high',
                    'value': event['spi_anomaly'],
                    'threshold': self.thresholds['climate_shock']
                })
            
            if event.get('temp_anomaly', 0) > self.thresholds['temperature_spike']:
                anomalies.append({
                    'type': 'temperature_spike',
                    'severity': 'medium',
                    'value': event['temp_anomaly'],
                    'threshold': self.thresholds['temperature_spike']
                })
        
        elif event['type'] == 'border_crossing':
            if event.get('count', 0) > self.thresholds['border_crossing_spike']:
                anomalies.append({
                    'type': 'border_crossing_spike',
                    'severity': 'high',
                    'value': event['count'],
                    'threshold': self.thresholds['border_crossing_spike']
                })
        
        return anomalies
    
    def calculate_baseline(self, event_type: str, hours: int = 24) -> float:
        """Calculate baseline from recent events."""
        recent = [e for e in self.recent_events[-hours*10:] if e.get('type') == event_type]
        if not recent:
            return 0.0
        
        if event_type == 'displacement':
            return np.mean([e.get('count', 0) for e in recent])
        elif event_type == 'conflict':
            return np.mean([e.get('intensity', 0) for e in recent])
        
        return 0.0
    
    def process_event_by_type(self, event: Dict[str, Any]):
        """Process events based on their type."""
        event_type = event.get('type')
        
        if event_type == 'displacement':
            self.update_displacement_metrics(event)
        elif event_type == 'conflict':
            self.update_conflict_metrics(event)
        elif event_type == 'climate':
            self.update_climate_metrics(event)
        elif event_type == 'border_crossing':
            self.update_border_metrics(event)
    
    def update_displacement_metrics(self, event: Dict[str, Any]):
        """Update displacement tracking metrics."""
        # Could integrate with existing models for real-time prediction updates
        logger.info(f"Processing displacement event: {event.get('count', 0)} people from {event.get('origin', 'unknown')}")
    
    def update_conflict_metrics(self, event: Dict[str, Any]):
        """Update conflict intensity metrics."""
        logger.info(f"Processing conflict event: intensity {event.get('intensity', 0)} at {event.get('location', 'unknown')}")
    
    def update_climate_metrics(self, event: Dict[str, Any]):
        """Update climate anomaly metrics."""
        logger.info(f"Processing climate event: SPI {event.get('spi_anomaly', 0)}, Temp {event.get('temp_anomaly', 0)}")
    
    def update_border_metrics(self, event: Dict[str, Any]):
        """Update border crossing metrics."""
        logger.info(f"Processing border crossing: {event.get('count', 0)} people at {event.get('border', 'unknown')}")
    
    def create_alert(self, event: Dict[str, Any], anomalies: list) -> Dict[str, Any]:
        """Create structured alert from event and anomalies."""
        alert = {
            'alert_id': f"mobility_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(event)) % 10000}",
            'timestamp': datetime.now().isoformat(),
            'event': event,
            'anomalies': anomalies,
            'severity': self.calculate_overall_severity(anomalies),
            'region': event.get('region', 'unknown'),
            'coordinates': event.get('coordinates', {}),
            'recommended_actions': self.generate_recommendations(anomalies)
        }
        
        return alert
    
    def calculate_overall_severity(self, anomalies: list) -> str:
        """Calculate overall alert severity."""
        if any(a.get('severity') == 'high' for a in anomalies):
            return 'high'
        elif any(a.get('severity') == 'medium' for a in anomalies):
            return 'medium'
        else:
            return 'low'
    
    def generate_recommendations(self, anomalies: list) -> list:
        """Generate actionable recommendations based on anomalies."""
        recommendations = []
        
        for anomaly in anomalies:
            if anomaly['type'] == 'sudden_movement':
                recommendations.append({
                    'action': 'Activate emergency response protocols',
                    'priority': 'immediate',
                    'target': 'humanitarian agencies'
                })
            elif anomaly['type'] == 'conflict_spike':
                recommendations.append({
                    'action': 'Monitor for potential mass displacement',
                    'priority': 'high',
                    'target': 'security and aid organizations'
                })
            elif anomaly['type'] == 'climate_shock':
                recommendations.append({
                    'action': 'Prepare drought/flood response measures',
                    'priority': 'medium',
                    'target': 'climate adaptation teams'
                })
        
        return recommendations
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert to multiple channels."""
        try:
            # Send to Kafka topic for downstream processing
            self.producer.send('mobility-alerts', value=alert)
            
            # Log alert
            logger.warning(f"ALERT {alert['severity'].upper()}: {alert['alert_id']}")
            logger.warning(f"Anomalies: {[a['type'] for a in alert['anomalies']]}")
            logger.warning(f"Recommendations: {[r['action'] for r in alert['recommended_actions']]}")
            
            # Could add webhook notifications, email alerts, etc.
            self.send_notification(alert)
            
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def send_notification(self, alert: Dict[str, Any]):
        """Send notification to external systems (webhooks, email, etc.)."""
        # Placeholder for notification integrations
        if alert['severity'] == 'high':
            logger.critical(f"CRITICAL ALERT: {alert['alert_id']} requires immediate attention")
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'consumer'):
            self.consumer.close()
        if hasattr(self, 'producer'):
            self.producer.close()
        logger.info("Stream processing cleanup completed")

# Utility functions for event generation and testing
def create_test_event(event_type: str, **kwargs) -> Dict[str, Any]:
    """Create test events for development and testing."""
    base_event = {
        'type': event_type,
        'timestamp': datetime.now().isoformat(),
        'region': kwargs.get('region', 'test_region'),
        'coordinates': kwargs.get('coordinates', {'lat': 0, 'lon': 0})
    }
    
    if event_type == 'displacement':
        base_event.update({
            'count': kwargs.get('count', 100),
            'origin': kwargs.get('origin', 'test_origin'),
            'destination': kwargs.get('destination', 'test_destination'),
            'demographics': kwargs.get('demographics', {})
        })
    elif event_type == 'conflict':
        base_event.update({
            'intensity': kwargs.get('intensity', 1.0),
            'location': kwargs.get('location', 'test_location'),
            'event_type': kwargs.get('event_type', 'violence'),
            'fatalities': kwargs.get('fatalities', 0)
        })
    elif event_type == 'climate':
        base_event.update({
            'spi_anomaly': kwargs.get('spi_anomaly', 0.0),
            'temp_anomaly': kwargs.get('temp_anomaly', 0.0),
            'precipitation': kwargs.get('precipitation', 0.0),
            'humidity': kwargs.get('humidity', 50.0)
        })
    elif event_type == 'border_crossing':
        base_event.update({
            'count': kwargs.get('count', 50),
            'border': kwargs.get('border', 'test_border'),
            'direction': kwargs.get('direction', 'inbound'),
            'mode': kwargs.get('mode', 'foot')
        })
    
    return base_event

if __name__ == "__main__":
    # Example usage
    processor = MobilityStreamProcessor()
    
    # For testing without Kafka
    test_events = [
        create_test_event('displacement', count=1500, region='horn_of_africa'),
        create_test_event('conflict', intensity=6.0, location='test_location'),
        create_test_event('climate', spi_anomaly=-2.5, temp_anomaly=4.0),
        create_test_event('border_crossing', count=600, border='test_border')
    ]
    
    print("Testing anomaly detection...")
    for event in test_events:
        anomalies = processor.detect_anomalies(event)
        if anomalies:
            print(f"Anomaly detected in {event['type']}: {anomalies}")
    
    # Uncomment to run actual stream processing
    # processor.process_stream()
