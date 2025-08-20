"""
Simulation database models
"""

from sqlalchemy import Column, String, Integer, Float, JSON, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from .base import Base, BaseModel

class Simulation(Base, BaseModel):
    """Simulation configuration and metadata"""
    
    __tablename__ = 'simulations'
    
    name = Column(String(255), nullable=False)
    simulation_type = Column(String(50), nullable=False)
    parameters = Column(JSON, nullable=False)
    status = Column(String(20), default='pending')
    description = Column(Text)
    
    # Relationships
    results = relationship("SimulationResult", back_populates="simulation")

class SimulationResult(Base, BaseModel):
    """Simulation results"""
    
    __tablename__ = 'simulation_results'
    
    simulation_id = Column(UUID(as_uuid=True), ForeignKey('simulations.id'), nullable=False)
    n_simulations = Column(Integer, nullable=False)
    n_steps = Column(Integer, nullable=False)
    
    # Results
    mean_final_value = Column(Float)
    std_final_value = Column(Float)
    min_final_value = Column(Float)
    max_final_value = Column(Float)
    
    # Percentiles
    percentile_5 = Column(Float)
    percentile_25 = Column(Float)
    percentile_50 = Column(Float)
    percentile_75 = Column(Float)
    percentile_95 = Column(Float)
    
    # Additional metrics
    var_95 = Column(Float)
    var_99 = Column(Float)
    expected_shortfall = Column(Float)
    
    # Metadata
    execution_time_seconds = Column(Float)
    convergence_achieved = Column(Boolean, default=False)
    
    # Raw results (stored as compressed JSON or file reference)
    results_data = Column(JSON)
    results_file_path = Column(String(500))
    
    # Relationships
    simulation = relationship("Simulation", back_populates="results")
