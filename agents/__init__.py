"""
Agents Package for Deep Reinforcement Learning
Contains DQN, A3C, and SAC agent implementations
"""

try:
    # Try normal imports first
    from .dqn_agent import DQNAgent
    from .a3c_agent import A3CAgent
    from .sac_agent import SACAgent
    print("✅ All agents imported successfully via normal imports")
except ImportError as e:
    print(f"⚠️ Normal import failed: {e}")
    print("🔧 Using alternative import method...")
    
    # Import DQN and A3C normally (these work)
    try:
        from .dqn_agent import DQNAgent
        from .a3c_agent import A3CAgent
        print("✅ DQN and A3C imported successfully")
    except ImportError as e:
        print(f"❌ Critical error importing DQN/A3C: {e}")
        raise
    
    # Use alternative method for SAC
    try:
        import importlib.util
        import os
        
        # Get the current directory of this __init__.py file
        current_dir = os.path.dirname(__file__)
        sac_path = os.path.join(current_dir, 'sac_agent.py')
        
        # Load SAC using importlib
        spec = importlib.util.spec_from_file_location("sac_agent", sac_path)
        sac_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sac_module)
        SACAgent = sac_module.SACAgent
        print("✅ SACAgent imported via alternative method")
        
    except Exception as sac_error:
        print(f"❌ SAC alternative import also failed: {sac_error}")
        # Define a dummy SACAgent to prevent import errors
        class SACAgent:
            def __init__(self, *args, **kwargs):
                raise NotImplementedError("SAC Agent is not available due to import issues")
        print("⚠️ Using dummy SACAgent class")

__all__ = ['DQNAgent', 'A3CAgent', 'SACAgent'] 