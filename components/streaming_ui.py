"""
Streaming UI Components for Real-time Processing Display

Provides Streamlit components for displaying intermediate processing steps
in real-time, similar to Claude Code's interface.
"""

import streamlit as st
import time
from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime

def create_streaming_container():
    """Create a container for streaming steps display."""
    return st.empty()

def render_streaming_step(container, step_data: Dict[str, Any], step_index: int):
    """
    Render a single streaming step with appropriate styling.
    
    Args:
        container: Streamlit container to render in
        step_data: Step data from StreamingStep
        step_index: Index of this step
    """
    
    # Status icons and colors
    status_config = {
        "starting": {"icon": "🔄", "color": "#FFA500"},
        "in_progress": {"icon": "⏳", "color": "#1E90FF"}, 
        "completed": {"icon": "✅", "color": "#32CD32"},
        "skipped": {"icon": "⏭️", "color": "#808080"},
        "error": {"icon": "❌", "color": "#FF4500"}
    }
    
    status = step_data.get("status", "starting")
    config = status_config.get(status, status_config["starting"])
    
    with container.container():
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
        
        with col1:
            st.markdown(f"<span style='font-size: 20px;'>{config['icon']}</span>", unsafe_allow_html=True)
        
        with col2:
            step_name = step_data.get("step_name", "Processing")
            message = step_data.get("message", "")
            
            # Main step display
            st.markdown(f"**{step_name}**")
            if message:
                st.markdown(f"<small style='color: #666;'>{message}</small>", unsafe_allow_html=True)
        
        with col3:
            # Duration display
            duration = step_data.get("duration")
            if duration is not None:
                st.markdown(f"<small>{duration:.2f}s</small>", unsafe_allow_html=True)
            elif status == "in_progress":
                st.markdown("<small>⏱️</small>", unsafe_allow_html=True)
        
        # Additional data display for completed steps
        if status == "completed" and step_data.get("data"):
            with st.expander("Details", expanded=False):
                st.json(step_data["data"])

def create_performance_controls():
    """Create UI controls for performance optimization."""
    
    st.subheader("⚡ Performance Controls")
    
    col1, col2 = st.columns(2)
    
    with col1:
        processing_mode = st.selectbox(
            "Processing Mode",
            options=["fast", "balanced", "comprehensive"],
            index=1,  # Default to balanced
            help="""
            **Fast**: ~0.5-1s, basic pattern matching, skip complex analysis
            **Balanced**: ~1-3s, optimized processing with key features  
            **Comprehensive**: ~3-8s, full agentic analysis and routing
            """,
            key="agentic_processing_mode"
        )
    
    with col2:
        enable_streaming = st.checkbox(
            "Enable Step Streaming",
            value=True,
            help="Show intermediate processing steps in real-time",
            key="enable_agentic_streaming"
        )
    
    # Advanced controls in expander
    with st.expander("Advanced Performance Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            hyde_timeout = st.slider(
                "HyDE Timeout (seconds)",
                min_value=1.0,
                max_value=10.0,
                value=3.0,
                step=0.5,
                key="hyde_timeout_seconds"
            )
            
            routing_timeout = st.slider(
                "Routing Timeout (seconds)", 
                min_value=0.5,
                max_value=5.0,
                value=2.0,
                step=0.5,
                key="routing_timeout_seconds"
            )
        
        with col2:
            skip_complex = st.checkbox(
                "Skip Complex Routing",
                value=False,
                help="Use simple pattern-based routing instead of LLM analysis",
                key="skip_complex_routing"
            )
            
            enable_parallel = st.checkbox(
                "Enable Parallel Processing",
                value=True,
                help="Process multiple components simultaneously when possible",
                key="enable_parallel_processing"
            )
    
    return {
        "processing_mode": processing_mode,
        "enable_streaming": enable_streaming,
        "hyde_timeout": hyde_timeout,
        "routing_timeout": routing_timeout,
        "skip_complex": skip_complex,
        "enable_parallel": enable_parallel
    }

class StreamingStepRenderer:
    """
    Class to manage streaming step rendering in Streamlit.
    
    Maintains state and provides smooth updates for real-time display.
    """
    
    def __init__(self):
        self.steps = []
        self.container = None
        self.step_containers = {}
    
    def initialize_display(self):
        """Initialize the streaming display area."""
        st.markdown("### 🔄 Processing Steps")
        self.container = st.container()
        return self.container
    
    def add_step(self, step_data: Dict[str, Any]):
        """Add a new step to the display."""
        step_id = f"{step_data.get('step_type', 'unknown')}_{len(self.steps)}"
        step_data['id'] = step_id
        self.steps.append(step_data)
        
        # Create container for this step
        if self.container:
            with self.container:
                step_container = st.empty()
                self.step_containers[step_id] = step_container
                render_streaming_step(step_container, step_data, len(self.steps) - 1)
    
    def update_step(self, step_id: str, step_data: Dict[str, Any]):
        """Update an existing step."""
        # Find and update step in list
        for i, step in enumerate(self.steps):
            if step.get('id') == step_id:
                self.steps[i].update(step_data)
                
                # Re-render the step
                if step_id in self.step_containers:
                    render_streaming_step(
                        self.step_containers[step_id], 
                        self.steps[i], 
                        i
                    )
                break
    
    def render_summary(self):
        """Render a summary of all completed steps."""
        if not self.steps:
            return
        
        completed_steps = [s for s in self.steps if s.get('status') == 'completed']
        total_time = sum(s.get('duration', 0) for s in completed_steps)
        
        with st.container():
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Steps", len(self.steps))
            
            with col2:
                st.metric("Completed", len(completed_steps))
            
            with col3:
                st.metric("Total Time", f"{total_time:.2f}s")
            
            with col4:
                error_count = len([s for s in self.steps if s.get('status') == 'error'])
                st.metric("Errors", error_count)

def create_performance_comparison():
    """Create a widget showing performance comparison between modes."""
    
    st.subheader("📊 Performance Comparison")
    
    performance_data = {
        "Mode": ["Fast", "Balanced", "Comprehensive"],
        "Avg Time": ["0.5-1s", "1-3s", "3-8s"],
        "Quality": ["Basic", "Good", "Excellent"],
        "Features": ["Pattern matching", "Optimized routing", "Full analysis"]
    }
    
    import pandas as pd
    df = pd.DataFrame(performance_data)
    
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )
    
    # Real-time performance metrics if available
    if "last_query_time" in st.session_state:
        st.info(f"⏱️ Last query processed in {st.session_state.last_query_time:.2f}s")

def show_processing_tips():
    """Show helpful tips for optimizing performance."""
    
    with st.expander("💡 Performance Tips"):
        st.markdown("""
        **For fastest results:**
        - Use "Fast" mode for simple queries
        - Disable step streaming if not needed
        - Skip complex routing for basic searches
        
        **For best quality:**
        - Use "Comprehensive" mode for complex queries
        - Enable all advanced features
        - Allow longer timeouts for detailed analysis
        
        **For balanced performance:**
        - Use "Balanced" mode (recommended)
        - Enable streaming for visual feedback
        - Keep default timeout settings
        """)

async def run_streaming_query(query_engine, query: str, step_renderer: StreamingStepRenderer):
    """
    Execute a streaming query and update the UI in real-time.
    
    This function should be called from the main Streamlit app to handle
    the async streaming query execution.
    """
    
    try:
        async for step in query_engine.query_streaming(query):
            step_data = {
                "step_type": step.step_type,
                "step_name": step.step_name,
                "status": step.status,
                "message": step.message,
                "data": step.data,
                "timestamp": step.timestamp,
                "duration": step.duration
            }
            
            step_renderer.add_step(step_data)
            
            # Small delay to make streaming visible
            await asyncio.sleep(0.1)
            
            # Get the final response when processing completes
            if step.step_type == "generation" and step.status == "completed":
                return step.data
                
    except Exception as e:
        step_renderer.add_step({
            "step_type": "error",
            "step_name": "Processing Error",
            "status": "error", 
            "message": f"Error: {str(e)}",
            "data": {"error": str(e)}
        })
        raise e

def display_mode_selector():
    """Create a prominent mode selector at the top of the page."""
    
    st.markdown("### ⚡ Query Processing Mode")
    
    modes = {
        "🚀 Fast": {
            "value": "fast",
            "description": "0.5-1s • Pattern matching • Best for simple queries",
            "color": "#FF6B6B"
        },
        "⚖️ Balanced": {
            "value": "balanced", 
            "description": "1-3s • Smart routing • Recommended for most queries",
            "color": "#4ECDC4"
        },
        "🔬 Comprehensive": {
            "value": "comprehensive",
            "description": "3-8s • Full analysis • Best for complex questions", 
            "color": "#45B7D1"
        }
    }
    
    selected_mode = st.radio(
        "Choose processing mode:",
        options=list(modes.keys()),
        index=1,  # Default to Balanced
        horizontal=True,
        key="processing_mode_selector"
    )
    
    # Show description for selected mode
    mode_info = modes[selected_mode]
    st.markdown(
        f"<div style='padding: 10px; background-color: {mode_info['color']}20; border-radius: 5px; margin-bottom: 20px;'>"
        f"<strong>{selected_mode}</strong>: {mode_info['description']}"
        f"</div>",
        unsafe_allow_html=True
    )
    
    return mode_info["value"]