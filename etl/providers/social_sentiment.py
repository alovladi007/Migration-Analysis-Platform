"""Social media sentiment analysis as early warning for migration patterns."""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
from collections import defaultdict

# Try to import social media and NLP libraries
try:
    import tweepy
    HAS_TWEEPY = True
except ImportError:
    HAS_TWEEPY = False
    print("Warning: tweepy not installed. Install with: pip install tweepy")

try:
    from transformers import pipeline
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Install with: pip install transformers")

try:
    from textblob import TextBlob
    HAS_TEXTBLOB = True
except ImportError:
    HAS_TEXTBLOB = False
    print("Warning: textblob not installed. Install with: pip install textblob")

logger = logging.getLogger(__name__)

@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    text: str
    sentiment_label: str  # 'positive', 'negative', 'neutral'
    sentiment_score: float  # -1 to 1
    confidence: float
    timestamp: str
    source: str
    location: Optional[Tuple[float, float]]

@dataclass
class MigrationSentimentSignal:
    """Migration-related sentiment signal."""
    signal_name: str
    sentiment_trend: str  # 'increasing_negative', 'decreasing_positive', 'stable'
    overall_sentiment: float
    volume: int
    confidence: float
    keywords: List[str]
    severity: str  # 'low', 'medium', 'high'
    potential_migration_impact: str

class SentimentMonitor:
    """Social media sentiment monitoring for migration patterns."""
    
    def __init__(self, 
                 twitter_bearer_token: str = None,
                 twitter_api_key: str = None,
                 twitter_api_secret: str = None):
        """
        Initialize sentiment monitor.
        
        Args:
            twitter_bearer_token: Twitter Bearer Token for API v2
            twitter_api_key: Twitter API Key
            twitter_api_secret: Twitter API Secret
        """
        self.twitter_client = None
        self.sentiment_classifier = None
        
        # Initialize Twitter client
        if HAS_TWEEPY and twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(
                    bearer_token=twitter_bearer_token,
                    consumer_key=twitter_api_key,
                    consumer_secret=twitter_api_secret,
                    wait_on_rate_limit=True
                )
                logger.info("Twitter client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
        else:
            logger.warning("Twitter client not available")
        
        # Initialize sentiment classifier
        if HAS_TRANSFORMERS:
            try:
                self.sentiment_classifier = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("Sentiment classifier initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize transformers classifier: {e}")
        
        # Migration-related keywords for different contexts
        self.migration_keywords = {
            'displacement': [
                'displaced', 'refugee', 'asylum', 'migration', 'fleeing',
                'forced migration', 'internal displacement', 'evacuation'
            ],
            'conflict': [
                'violence', 'conflict', 'war', 'fighting', 'attack', 'bombing',
                'military', 'armed groups', 'terrorism', 'insecurity'
            ],
            'climate': [
                'drought', 'flood', 'climate change', 'extreme weather',
                'crop failure', 'famine', 'food shortage', 'water shortage'
            ],
            'economic': [
                'unemployment', 'poverty', 'economic crisis', 'inflation',
                'food prices', 'job loss', 'economic hardship'
            ],
            'border': [
                'border crossing', 'immigration', 'deportation', 'border control',
                'illegal crossing', 'migration route', 'smuggling'
            ]
        }
        
        # Regional keywords
        self.regional_keywords = {
            'horn_of_africa': ['ethiopia', 'kenya', 'somalia', 'djibouti', 'eritrea', 'sudan'],
            'sahel': ['mali', 'niger', 'burkina faso', 'chad', 'mauritania', 'nigeria'],
            'central_america': ['guatemala', 'honduras', 'el salvador', 'mexico', 'belize'],
            'south_asia': ['afghanistan', 'pakistan', 'bangladesh', 'india', 'sri lanka']
        }
    
    def analyze_migration_sentiment(self, 
                                  region: str,
                                  keywords: List[str] = None,
                                  max_tweets: int = 100,
                                  time_window_hours: int = 24) -> List[SentimentResult]:
        """
        Analyze social media sentiment about migration in a region.
        
        Args:
            region: Region identifier
            keywords: Custom keywords (optional)
            max_tweets: Maximum number of tweets to analyze
            time_window_hours: Time window for analysis
            
        Returns:
            List of sentiment results
        """
        if not self.twitter_client:
            logger.warning("Twitter client not available, using mock sentiment data")
            return self._mock_sentiment_analysis(region, max_tweets)
        
        logger.info(f"Analyzing migration sentiment for region: {region}")
        
        # Build search query
        query_terms = []
        
        # Add region-specific keywords
        if region in self.regional_keywords:
            query_terms.extend(self.regional_keywords[region])
        
        # Add migration keywords
        if keywords:
            query_terms.extend(keywords)
        else:
            # Use default migration keywords
            for category, terms in self.migration_keywords.items():
                query_terms.extend(terms[:3])  # Use first 3 terms from each category
        
        # Create search query
        query = " OR ".join(query_terms[:10])  # Limit to 10 terms
        query += " lang:en -is:retweet"  # English only, no retweets
        
        logger.debug(f"Search query: {query}")
        
        try:
            # Search for tweets
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),  # Twitter API limit
                tweet_fields=['created_at', 'geo', 'public_metrics'],
                expansions=['geo.place_id']
            )
            
            if not tweets.data:
                logger.warning("No tweets found for the query")
                return []
            
            # Analyze sentiment for each tweet
            sentiment_results = []
            
            for tweet in tweets.data:
                sentiment_result = self._analyze_tweet_sentiment(tweet)
                if sentiment_result:
                    sentiment_results.append(sentiment_result)
            
            logger.info(f"Analyzed sentiment for {len(sentiment_results)} tweets")
            return sentiment_results
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return self._mock_sentiment_analysis(region, max_tweets)
    
    def _analyze_tweet_sentiment(self, tweet) -> Optional[SentimentResult]:
        """Analyze sentiment of a single tweet."""
        try:
            text = tweet.text
            
            # Clean text
            text = self._clean_text(text)
            
            if len(text) < 10:  # Skip very short tweets
                return None
            
            # Get sentiment
            sentiment_label, sentiment_score, confidence = self._get_sentiment(text)
            
            # Get location if available
            location = None
            if hasattr(tweet, 'geo') and tweet.geo:
                if 'coordinates' in tweet.geo:
                    coords = tweet.geo['coordinates']
                    location = (coords['coordinates'][1], coords['coordinates'][0])  # lat, lon
            
            return SentimentResult(
                text=text[:200],  # Truncate for storage
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                confidence=confidence,
                timestamp=tweet.created_at.isoformat() if tweet.created_at else datetime.now().isoformat(),
                source='twitter',
                location=location
            )
            
        except Exception as e:
            logger.error(f"Error analyzing tweet sentiment: {e}")
            return None
    
    def _get_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Get sentiment analysis for text."""
        if self.sentiment_classifier:
            try:
                results = self.sentiment_classifier(text)
                
                # Find highest scoring sentiment
                best_result = max(results[0], key=lambda x: x['score'])
                
                label = best_result['label']
                score = best_result['score']
                confidence = score
                
                # Convert to standard format
                if label == 'LABEL_0':  # Negative
                    sentiment_score = -score
                    sentiment_label = 'negative'
                elif label == 'LABEL_1':  # Neutral
                    sentiment_score = 0
                    sentiment_label = 'neutral'
                else:  # Positive
                    sentiment_score = score
                    sentiment_label = 'positive'
                
                return sentiment_label, sentiment_score, confidence
                
            except Exception as e:
                logger.error(f"Error in transformers sentiment analysis: {e}")
        
        # Fallback to TextBlob
        if HAS_TEXTBLOB:
            try:
                blob = TextBlob(text)
                sentiment_score = blob.sentiment.polarity
                
                if sentiment_score > 0.1:
                    sentiment_label = 'positive'
                elif sentiment_score < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                confidence = abs(sentiment_score)
                
                return sentiment_label, sentiment_score, confidence
                
            except Exception as e:
                logger.error(f"Error in TextBlob sentiment analysis: {e}")
        
        # Final fallback: simple keyword-based sentiment
        return self._simple_sentiment_analysis(text)
    
    def _simple_sentiment_analysis(self, text: str) -> Tuple[str, float, float]:
        """Simple keyword-based sentiment analysis."""
        positive_words = ['good', 'great', 'help', 'support', 'safe', 'better', 'hope', 'peace']
        negative_words = ['bad', 'terrible', 'crisis', 'danger', 'violence', 'war', 'fear', 'hunger']
        
        text_lower = text.lower()
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if negative_count > positive_count:
            sentiment_label = 'negative'
            sentiment_score = -0.5
        elif positive_count > negative_count:
            sentiment_label = 'positive'
            sentiment_score = 0.5
        else:
            sentiment_label = 'neutral'
            sentiment_score = 0.0
        
        confidence = 0.3  # Low confidence for simple analysis
        
        return sentiment_label, sentiment_score, confidence
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess text."""
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags (but keep the text)
        text = re.sub(r'#', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def generate_migration_sentiment_signals(self, 
                                           sentiment_results: List[SentimentResult],
                                           region: str,
                                           time_window_hours: int = 24) -> List[MigrationSentimentSignal]:
        """
        Generate migration-related sentiment signals.
        
        Args:
            sentiment_results: List of sentiment analysis results
            region: Region identifier
            time_window_hours: Time window for analysis
            
        Returns:
            List of migration sentiment signals
        """
        if not sentiment_results:
            return []
        
        logger.info(f"Generating migration sentiment signals for {len(sentiment_results)} results")
        
        signals = []
        
        # Calculate overall sentiment metrics
        sentiment_scores = [r.sentiment_score for r in sentiment_results]
        overall_sentiment = np.mean(sentiment_scores)
        
        # Analyze sentiment by category
        category_sentiments = defaultdict(list)
        
        for result in sentiment_results:
            text_lower = result.text.lower()
            
            for category, keywords in self.migration_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    category_sentiments[category].append(result.sentiment_score)
        
        # Generate signals for each category
        for category, scores in category_sentiments.items():
            if len(scores) >= 5:  # Minimum threshold
                category_sentiment = np.mean(scores)
                volume = len(scores)
                
                # Determine trend (simplified)
                if category_sentiment < -0.3:
                    trend = 'increasing_negative'
                    severity = 'high'
                elif category_sentiment < -0.1:
                    trend = 'stable_negative'
                    severity = 'medium'
                elif category_sentiment > 0.3:
                    trend = 'increasing_positive'
                    severity = 'low'
                else:
                    trend = 'stable_neutral'
                    severity = 'low'
                
                # Determine migration impact
                if category in ['displacement', 'conflict'] and category_sentiment < -0.2:
                    migration_impact = 'positive'  # Negative sentiment about displacement/conflict may indicate migration
                elif category in ['economic', 'climate'] and category_sentiment < -0.3:
                    migration_impact = 'positive'  # Economic/climate stress may drive migration
                else:
                    migration_impact = 'neutral'
                
                signal = MigrationSentimentSignal(
                    signal_name=f"{region}_{category}_sentiment",
                    sentiment_trend=trend,
                    overall_sentiment=category_sentiment,
                    volume=volume,
                    confidence=min(0.9, volume / 50),  # Confidence based on volume
                    keywords=self.migration_keywords[category],
                    severity=severity,
                    potential_migration_impact=migration_impact
                )
                
                signals.append(signal)
        
        # Overall regional sentiment signal
        if len(sentiment_scores) >= 10:
            if overall_sentiment < -0.2:
                trend = 'increasing_negative'
                severity = 'high'
                migration_impact = 'positive'
            elif overall_sentiment < 0:
                trend = 'stable_negative'
                severity = 'medium'
                migration_impact = 'neutral'
            else:
                trend = 'stable_positive'
                severity = 'low'
                migration_impact = 'negative'
            
            overall_signal = MigrationSentimentSignal(
                signal_name=f"{region}_overall_sentiment",
                sentiment_trend=trend,
                overall_sentiment=overall_sentiment,
                volume=len(sentiment_results),
                confidence=min(0.9, len(sentiment_results) / 100),
                keywords=[],
                severity=severity,
                potential_migration_impact=migration_impact
            )
            
            signals.append(overall_signal)
        
        logger.info(f"Generated {len(signals)} migration sentiment signals")
        return signals
    
    def _mock_sentiment_analysis(self, region: str, max_tweets: int) -> List[SentimentResult]:
        """Generate mock sentiment analysis results for testing."""
        logger.info(f"Generating mock sentiment data for {region}")
        
        mock_results = []
        
        # Sample migration-related texts
        sample_texts = [
            "Refugees fleeing conflict in the region need urgent help",
            "Economic crisis forcing families to migrate",
            "Drought conditions worsening, people leaving their homes",
            "Border crossing becoming more dangerous",
            "Community helping displaced families",
            "Food shortage affecting entire region",
            "Violence escalating, many seeking safety elsewhere",
            "Climate change making life unbearable",
            "Government providing aid to migrants",
            "Migration routes becoming more difficult"
        ]
        
        sentiment_labels = ['negative', 'negative', 'negative', 'negative', 'positive', 
                          'negative', 'negative', 'negative', 'positive', 'negative']
        
        for i in range(min(max_tweets, len(sample_texts))):
            text = sample_texts[i]
            sentiment_label = sentiment_labels[i]
            
            # Generate realistic sentiment score
            if sentiment_label == 'negative':
                sentiment_score = np.random.uniform(-0.8, -0.2)
            elif sentiment_label == 'positive':
                sentiment_score = np.random.uniform(0.2, 0.8)
            else:
                sentiment_score = np.random.uniform(-0.1, 0.1)
            
            result = SentimentResult(
                text=text,
                sentiment_label=sentiment_label,
                sentiment_score=sentiment_score,
                confidence=np.random.uniform(0.6, 0.9),
                timestamp=(datetime.now() - timedelta(hours=np.random.randint(0, 24))).isoformat(),
                source='twitter_mock',
                location=None
            )
            
            mock_results.append(result)
        
        return mock_results

def integrate_sentiment_with_flows(sentiment_signals: List[MigrationSentimentSignal],
                                 flow_data: pd.DataFrame) -> pd.DataFrame:
    """Integrate sentiment signals with migration flow data."""
    enhanced_flows = flow_data.copy()
    
    # Add sentiment indicators
    enhanced_flows['sentiment_negative_ratio'] = 0.0
    enhanced_flows['sentiment_volume'] = 0
    enhanced_flows['sentiment_confidence'] = 0.0
    enhanced_flows['migration_sentiment_signal'] = 'neutral'
    
    # Aggregate sentiment signals by region
    if sentiment_signals:
        # Calculate weighted average sentiment
        total_volume = sum(signal.volume for signal in sentiment_signals)
        weighted_sentiment = sum(signal.overall_sentiment * signal.volume for signal in sentiment_signals) / total_volume if total_volume > 0 else 0
        
        # Calculate negative ratio
        negative_signals = [s for s in sentiment_signals if s.overall_sentiment < -0.1]
        negative_ratio = len(negative_signals) / len(sentiment_signals) if sentiment_signals else 0
        
        # Set values for all flows (simplified)
        enhanced_flows['sentiment_negative_ratio'] = negative_ratio
        enhanced_flows['sentiment_volume'] = total_volume
        enhanced_flows['sentiment_confidence'] = np.mean([s.confidence for s in sentiment_signals])
        
        # Determine migration signal
        if weighted_sentiment < -0.3:
            enhanced_flows['migration_sentiment_signal'] = 'high_risk'
        elif weighted_sentiment < -0.1:
            enhanced_flows['migration_sentiment_signal'] = 'medium_risk'
        else:
            enhanced_flows['migration_sentiment_signal'] = 'low_risk'
    
    return enhanced_flows

if __name__ == "__main__":
    # Test sentiment monitoring
    print("Testing social media sentiment monitoring...")
    
    # Create monitor
    monitor = SentimentMonitor()
    
    # Test sentiment analysis
    sentiment_results = monitor.analyze_migration_sentiment(
        region='horn_of_africa',
        max_tweets=50
    )
    
    print(f"Analyzed sentiment for {len(sentiment_results)} posts")
    
    # Test signal generation
    signals = monitor.generate_migration_sentiment_signals(
        sentiment_results=sentiment_results,
        region='horn_of_africa'
    )
    
    print(f"Generated {len(signals)} sentiment signals")
    
    for signal in signals:
        print(f"  - {signal.signal_name}: {signal.overall_sentiment:.3f} ({signal.severity})")
        print(f"    Trend: {signal.sentiment_trend}, Impact: {signal.potential_migration_impact}")
    
    print("Social media sentiment monitoring test completed!")
