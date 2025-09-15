"""Agent-based model for individual migration decision-making simulation."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from collections import defaultdict
import random
from datetime import datetime, timedelta

# Try to import Mesa for agent-based modeling
try:
    import mesa
    from mesa import Agent, Model
    from mesa.time import RandomActivation
    from mesa.space import MultiGrid, ContinuousSpace
    from mesa.datacollection import DataCollector
    HAS_MESA = True
except ImportError:
    HAS_MESA = False
    print("Warning: mesa not installed. Install with: pip install mesa")

logger = logging.getLogger(__name__)

@dataclass
class LocationData:
    """Data for a geographical location."""
    location_id: str
    coordinates: Tuple[float, float]
    population: int
    conflict_level: float
    drought_severity: float
    unemployment_rate: float
    access_to_services: float
    economic_opportunity: float
    climate_stress: float

@dataclass
class MigrationDecision:
    """Result of migration decision process."""
    agent_id: str
    origin: str
    destination: str
    decision: bool  # True if migrating
    utility: float
    push_factors: float
    pull_factors: float
    migration_cost: float
    timestamp: str

class MigrantAgent(Agent):
    """Individual agent representing a potential migrant."""
    
    def __init__(self, unique_id: str, model, origin: str, resources: float = 100.0):
        """
        Initialize migrant agent.
        
        Args:
            unique_id: Unique identifier
            model: Reference to the model
            origin: Origin location ID
            resources: Available resources (0-200)
        """
        super().__init__(unique_id, model)
        self.origin = origin
        self.location = origin
        self.resources = resources
        self.risk_tolerance = np.random.beta(2, 5)  # Most agents are risk-averse
        self.network_size = np.random.poisson(5)  # Social network size
        self.education_level = np.random.uniform(0, 1)  # 0-1 scale
        self.age = np.random.uniform(18, 65)  # Working age
        self.family_size = np.random.poisson(3)  # Family members
        self.has_destination_info = False
        self.migration_history = []
        self.decisions_made = 0
        self.max_decisions = 10  # Prevent infinite loops
        
        # Migration preferences (weights for different factors)
        self.preferences = {
            'economic': np.random.uniform(0.3, 0.8),
            'safety': np.random.uniform(0.2, 0.9),
            'family': np.random.uniform(0.1, 0.7),
            'climate': np.random.uniform(0.1, 0.6)
        }
        
        logger.debug(f"Created agent {unique_id} at {origin} with {resources:.1f} resources")
    
    def step(self):
        """Execute one step of agent behavior."""
        if self.decisions_made >= self.max_decisions:
            return
        
        # Get current location data
        location_data = self.model.get_location_data(self.location)
        if not location_data:
            return
        
        # Evaluate migration decision
        migration_decision = self.evaluate_migration_decision(location_data)
        
        if migration_decision.decision:
            self.migrate(migration_decision)
        
        self.decisions_made += 1
    
    def evaluate_migration_decision(self, current_location: LocationData) -> MigrationDecision:
        """Evaluate whether to migrate based on current conditions."""
        # Calculate push factors (reasons to leave current location)
        push_factors = self.calculate_push_factors(current_location)
        
        # Get available destinations
        destinations = self.model.get_available_destinations(self.location)
        
        if not destinations:
            # No destinations available
            return MigrationDecision(
                agent_id=self.unique_id,
                origin=self.location,
                destination=self.location,
                decision=False,
                utility=0.0,
                push_factors=push_factors,
                pull_factors=0.0,
                migration_cost=0.0,
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate utilities for each destination
        destination_utilities = {}
        
        for dest_id in destinations:
            dest_data = self.model.get_location_data(dest_id)
            if dest_data:
                pull_factors = self.calculate_pull_factors(dest_data)
                migration_cost = self.calculate_migration_cost(dest_id)
                
                # Calculate utility
                utility = self.calculate_utility(push_factors, pull_factors, migration_cost)
                destination_utilities[dest_id] = {
                    'utility': utility,
                    'pull_factors': pull_factors,
                    'migration_cost': migration_cost
                }
        
        # Choose best destination
        if destination_utilities:
            best_dest = max(destination_utilities.keys(), 
                          key=lambda x: destination_utilities[x]['utility'])
            best_utility = destination_utilities[best_dest]['utility']
            
            # Migration threshold based on risk tolerance and resources
            migration_threshold = self.risk_tolerance * 10 + (self.resources / 20)
            
            decision = best_utility > migration_threshold
            
            return MigrationDecision(
                agent_id=self.unique_id,
                origin=self.location,
                destination=best_dest,
                decision=decision,
                utility=best_utility,
                push_factors=push_factors,
                pull_factors=destination_utilities[best_dest]['pull_factors'],
                migration_cost=destination_utilities[best_dest]['migration_cost'],
                timestamp=datetime.now().isoformat()
            )
        else:
            return MigrationDecision(
                agent_id=self.unique_id,
                origin=self.location,
                destination=self.location,
                decision=False,
                utility=0.0,
                push_factors=push_factors,
                pull_factors=0.0,
                migration_cost=0.0,
                timestamp=datetime.now().isoformat()
            )
    
    def calculate_push_factors(self, location_data: LocationData) -> float:
        """Calculate push factors (reasons to leave current location)."""
        push_score = 0.0
        
        # Conflict factor
        conflict_push = location_data.conflict_level * self.preferences['safety'] * 2.0
        push_score += conflict_push
        
        # Economic factor
        economic_push = (1 - location_data.economic_opportunity) * self.preferences['economic'] * 1.5
        push_score += economic_push
        
        # Climate factor
        climate_push = location_data.drought_severity * self.preferences['climate'] * 1.2
        push_score += climate_push
        
        # Unemployment factor
        unemployment_push = location_data.unemployment_rate * self.preferences['economic'] * 1.0
        push_score += unemployment_push
        
        # Access to services factor
        services_push = (1 - location_data.access_to_services) * 0.8
        push_score += services_push
        
        return min(push_score, 10.0)  # Cap at 10
    
    def calculate_pull_factors(self, location_data: LocationData) -> float:
        """Calculate pull factors (reasons to move to destination)."""
        pull_score = 0.0
        
        # Economic opportunity
        economic_pull = location_data.economic_opportunity * self.preferences['economic'] * 1.5
        pull_score += economic_pull
        
        # Safety (inverse of conflict)
        safety_pull = (1 - location_data.conflict_level) * self.preferences['safety'] * 2.0
        pull_score += safety_pull
        
        # Access to services
        services_pull = location_data.access_to_services * 0.8
        pull_score += services_pull
        
        # Network effect (if agent has information about destination)
        if self.has_destination_info:
            network_pull = min(self.network_size / 10, 1.0) * 0.5
            pull_score += network_pull
        
        return min(pull_score, 10.0)  # Cap at 10
    
    def calculate_migration_cost(self, destination: str) -> float:
        """Calculate cost of migration to destination."""
        # Distance-based cost
        origin_coords = self.model.locations[self.location].coordinates
        dest_coords = self.model.locations[destination].coordinates
        
        distance = np.sqrt((dest_coords[0] - origin_coords[0])**2 + 
                          (dest_coords[1] - origin_coords[1])**2)
        
        distance_cost = distance * 0.5  # Cost per distance unit
        
        # Resource-based cost (agents with more resources can afford better migration)
        resource_cost = max(0, 10 - self.resources / 10)
        
        # Family size increases cost
        family_cost = self.family_size * 2
        
        # Age factor (younger people have lower migration costs)
        age_factor = max(0.5, 1 - (self.age - 18) / 47)
        
        total_cost = (distance_cost + resource_cost + family_cost) * age_factor
        
        return total_cost
    
    def calculate_utility(self, push_factors: float, pull_factors: float, migration_cost: float) -> float:
        """Calculate utility of migration decision."""
        # Utility = pull factors + push factors - migration cost
        utility = pull_factors + push_factors - migration_cost
        
        # Add some randomness
        utility += np.random.normal(0, 0.5)
        
        return utility
    
    def migrate(self, migration_decision: MigrationDecision):
        """Execute migration to destination."""
        if migration_decision.decision:
            old_location = self.location
            self.location = migration_decision.destination
            
            # Update resources (migration costs resources)
            self.resources -= migration_decision.migration_cost
            self.resources = max(0, self.resources)
            
            # Record migration
            self.migration_history.append({
                'from': old_location,
                'to': migration_decision.destination,
                'timestamp': migration_decision.timestamp,
                'utility': migration_decision.utility
            })
            
            # Update information about destination
            self.has_destination_info = True
            
            # Notify model of migration
            self.model.record_migration(migration_decision)
            
            logger.debug(f"Agent {self.unique_id} migrated from {old_location} to {migration_decision.destination}")

class MigrationABM(Model):
    """Agent-based model for migration simulation."""
    
    def __init__(self, 
                 n_agents: int = 1000,
                 width: int = 50,
                 height: int = 50,
                 climate_scenario: str = 'baseline',
                 locations_data: Dict[str, LocationData] = None):
        """
        Initialize migration ABM.
        
        Args:
            n_agents: Number of agents
            width: Grid width
            height: Grid height
            climate_scenario: Climate scenario ('baseline', 'drought', 'conflict')
            locations_data: Dictionary of location data
        """
        if not HAS_MESA:
            raise RuntimeError("Mesa not installed. Install with: pip install mesa")
        
        super().__init__()
        self.num_agents = n_agents
        self.climate_scenario = climate_scenario
        
        # Initialize space (continuous space for geographical coordinates)
        self.space = ContinuousSpace(width, height, True)
        
        # Initialize scheduler
        self.schedule = RandomActivation(self)
        
        # Location data
        self.locations = locations_data or self._create_default_locations()
        
        # Migration tracking
        self.migrations = []
        self.migration_flows = defaultdict(int)
        
        # Create agents
        self._create_agents()
        
        # Initialize data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Total_Displaced": self.count_displaced,
                "Migration_Rate": self.calculate_migration_rate,
                "Average_Utility": self.calculate_average_utility,
                "Active_Agents": lambda m: m.schedule.get_agent_count()
            },
            agent_reporters={
                "Location": "location",
                "Resources": "resources",
                "Migration_Count": lambda a: len(a.migration_history)
            }
        )
        
        logger.info(f"Initialized ABM with {n_agents} agents and {len(self.locations)} locations")
    
    def _create_default_locations(self) -> Dict[str, LocationData]:
        """Create default location data for simulation."""
        locations = {}
        
        # Create 5 default locations
        location_configs = [
            ("A", (10, 10), 100000, 0.2, 0.3, 0.1, 0.7, 0.6, 0.4),
            ("B", (20, 15), 150000, 0.1, 0.2, 0.05, 0.8, 0.7, 0.3),
            ("C", (15, 25), 200000, 0.5, 0.6, 0.2, 0.5, 0.4, 0.7),
            ("D", (30, 20), 120000, 0.3, 0.4, 0.15, 0.6, 0.5, 0.5),
            ("E", (25, 35), 180000, 0.4, 0.5, 0.18, 0.55, 0.45, 0.6)
        ]
        
        for config in location_configs:
            location_id, coords, pop, conflict, drought, unemployment, access, economic, climate = config
            
            # Apply climate scenario
            if self.climate_scenario == 'drought':
                drought *= 1.5
                climate *= 1.3
            elif self.climate_scenario == 'conflict':
                conflict *= 1.5
            
            locations[location_id] = LocationData(
                location_id=location_id,
                coordinates=coords,
                population=pop,
                conflict_level=conflict,
                drought_severity=drought,
                unemployment_rate=unemployment,
                access_to_services=access,
                economic_opportunity=economic,
                climate_stress=climate
            )
        
        return locations
    
    def _create_agents(self):
        """Create and place agents in the model."""
        for i in range(self.num_agents):
            # Assign agent to random location
            origin = random.choice(list(self.locations.keys()))
            
            # Create agent
            agent = MigrantAgent(i, self, origin)
            
            # Add to schedule
            self.schedule.add(agent)
            
            # Place in space
            coords = self.locations[origin].coordinates
            self.space.place_agent(agent, coords)
    
    def step(self):
        """Execute one step of the model."""
        # Apply climate/conflict scenarios
        self._apply_scenario_effects()
        
        # Execute agent steps
        self.schedule.step()
        
        # Collect data
        self.datacollector.collect(self)
        
        # Update step counter
        self.step_count += 1
    
    def _apply_scenario_effects(self):
        """Apply climate and conflict scenario effects."""
        if self.climate_scenario == 'drought':
            # Increase drought severity over time
            for location in self.locations.values():
                location.drought_severity = min(1.0, location.drought_severity + 0.01)
                location.climate_stress = min(1.0, location.climate_stress + 0.005)
        
        elif self.climate_scenario == 'conflict':
            # Increase conflict levels
            for location in self.locations.values():
                location.conflict_level = min(1.0, location.conflict_level + 0.02)
    
    def get_location_data(self, location_id: str) -> Optional[LocationData]:
        """Get data for a specific location."""
        return self.locations.get(location_id)
    
    def get_available_destinations(self, origin: str) -> List[str]:
        """Get available destinations from origin."""
        # All locations except origin
        return [loc_id for loc_id in self.locations.keys() if loc_id != origin]
    
    def record_migration(self, migration_decision: MigrationDecision):
        """Record migration in the model."""
        self.migrations.append(migration_decision)
        
        # Update flow counts
        flow_key = f"{migration_decision.origin}->{migration_decision.destination}"
        self.migration_flows[flow_key] += 1
    
    def count_displaced(self) -> int:
        """Count total number of displaced agents."""
        displaced = 0
        for agent in self.schedule.agents:
            if len(agent.migration_history) > 0:
                displaced += 1
        return displaced
    
    def calculate_migration_rate(self) -> float:
        """Calculate current migration rate."""
        if self.num_agents == 0:
            return 0.0
        return self.count_displaced() / self.num_agents
    
    def calculate_average_utility(self) -> float:
        """Calculate average utility of recent migration decisions."""
        if not self.migrations:
            return 0.0
        
        recent_migrations = self.migrations[-100:]  # Last 100 migrations
        utilities = [m.utility for m in recent_migrations]
        return np.mean(utilities) if utilities else 0.0
    
    def get_migration_flows(self) -> Dict[str, int]:
        """Get current migration flows."""
        return dict(self.migration_flows)
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics."""
        agents = list(self.schedule.agents)
        
        if not agents:
            return {}
        
        return {
            'total_agents': len(agents),
            'displaced_agents': self.count_displaced(),
            'average_resources': np.mean([a.resources for a in agents]),
            'average_age': np.mean([a.age for a in agents]),
            'average_family_size': np.mean([a.family_size for a in agents]),
            'average_risk_tolerance': np.mean([a.risk_tolerance for a in agents]),
            'total_migrations': len(self.migrations),
            'migration_rate': self.calculate_migration_rate(),
            'average_utility': self.calculate_average_utility()
        }

def run_migration_simulation(n_agents: int = 1000,
                           n_steps: int = 100,
                           climate_scenario: str = 'baseline',
                           locations_data: Dict[str, LocationData] = None) -> Dict[str, Any]:
    """
    Run a complete migration simulation.
    
    Args:
        n_agents: Number of agents
        n_steps: Number of simulation steps
        climate_scenario: Climate scenario
        locations_data: Location data
        
    Returns:
        Simulation results
    """
    if not HAS_MESA:
        logger.error("Mesa not available for agent-based simulation")
        return {}
    
    logger.info(f"Starting migration simulation with {n_agents} agents for {n_steps} steps")
    
    # Create and run model
    model = MigrationABM(
        n_agents=n_agents,
        climate_scenario=climate_scenario,
        locations_data=locations_data
    )
    
    # Run simulation
    for step in range(n_steps):
        model.step()
        
        if step % 20 == 0:
            logger.info(f"Step {step}: {model.count_displaced()} displaced agents")
    
    # Collect results
    results = {
        'model_data': model.datacollector.get_model_vars_dataframe(),
        'agent_data': model.datacollector.get_agent_vars_dataframe(),
        'migration_flows': model.get_migration_flows(),
        'final_statistics': model.get_agent_statistics(),
        'total_migrations': len(model.migrations),
        'scenario': climate_scenario
    }
    
    logger.info(f"Simulation completed: {model.count_displaced()} agents displaced")
    
    return results

def analyze_simulation_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze and summarize simulation results."""
    if not results:
        return {}
    
    model_data = results.get('model_data', pd.DataFrame())
    agent_data = results.get('agent_data', pd.DataFrame())
    migration_flows = results.get('migration_flows', {})
    
    analysis = {
        'scenario': results.get('scenario', 'unknown'),
        'total_migrations': results.get('total_migrations', 0),
        'final_displaced': model_data['Total_Displaced'].iloc[-1] if not model_data.empty else 0,
        'final_migration_rate': model_data['Migration_Rate'].iloc[-1] if not model_data.empty else 0,
        'average_utility': model_data['Average_Utility'].iloc[-1] if not model_data.empty else 0,
        'top_migration_corridors': dict(sorted(migration_flows.items(), key=lambda x: x[1], reverse=True)[:5])
    }
    
    # Time series analysis
    if not model_data.empty:
        analysis['migration_rate_trend'] = model_data['Migration_Rate'].iloc[-1] - model_data['Migration_Rate'].iloc[0]
        analysis['peak_migration_step'] = model_data['Total_Displaced'].idxmax()
    
    return analysis

if __name__ == "__main__":
    # Test agent-based modeling
    print("Testing agent-based migration simulation...")
    
    if HAS_MESA:
        # Run baseline simulation
        results_baseline = run_migration_simulation(
            n_agents=500,
            n_steps=50,
            climate_scenario='baseline'
        )
        
        print(f"Baseline simulation: {results_baseline['total_migrations']} total migrations")
        
        # Run drought scenario
        results_drought = run_migration_simulation(
            n_agents=500,
            n_steps=50,
            climate_scenario='drought'
        )
        
        print(f"Drought simulation: {results_drought['total_migrations']} total migrations")
        
        # Analyze results
        baseline_analysis = analyze_simulation_results(results_baseline)
        drought_analysis = analyze_simulation_results(results_drought)
        
        print(f"Baseline migration rate: {baseline_analysis['final_migration_rate']:.3f}")
        print(f"Drought migration rate: {drought_analysis['final_migration_rate']:.3f}")
        
        print("Top migration corridors (drought scenario):")
        for corridor, count in drought_analysis['top_migration_corridors'].items():
            print(f"  {corridor}: {count} migrations")
    
    else:
        print("Mesa not available - skipping agent-based simulation test")
    
    print("Agent-based modeling test completed!")
