"""
Quick Test Script for SAC Implementation
Verify that Qi's SAC agent can run properly
"""

import numpy as np
import random
from pathlib import Path

# Test environment loading
try:
    from world.environment_qi import RestaurantDeliveryEnvironment
    from agents.sac_agent import SACAgent
    print("✅ Successfully imported SAC agent and environment")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def test_sac_basic():
    """Test basic SAC functionality"""
    print("\n🧪 Testing SAC Basic Functionality...")
    
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    env = RestaurantDeliveryEnvironment(
        space_file=Path("grid_configs/my_first_restaurant.npz"), 
        enable_gui=False,
        max_episode_steps=100
    )
    
    # Initialize SAC agent
    agent = SACAgent(
        state_size=10,
        action_size=5,
        lr=1e-3,  # Higher lr for faster testing
        gamma=0.99,
        tau=0.01,  # Faster updates for testing
        alpha=0.2,
        buffer_size=1000,  # Smaller buffer for testing
        batch_size=32,     # Smaller batch for testing
        hidden_size=128,   # Smaller network for testing
        auto_temp=True
    )
    
    print(f"✅ SAC agent initialized with device: {agent.device}")
    
    # Test state conversion
    obs = env.reset()
    state_vector = agent.state_to_vector(obs)
    
    print(f"✅ State conversion works: {state_vector.shape} -> {state_vector[:3]}")
    assert state_vector.shape == (10,), f"Expected state shape (10,), got {state_vector.shape}"
    
    # Test action selection
    action = agent.act(obs)
    print(f"✅ Action selection works: {action} (type: {type(action)})")
    assert isinstance(action, tuple) and len(action) == 2, f"Action {action} should be tuple of length 2"
    velocity, rotation = action
    assert -1.0 <= velocity <= 1.0, f"Velocity {velocity} out of valid range [-1.0, 1.0]"
    assert -1.0 <= rotation <= 1.0, f"Rotation {rotation} out of valid range [-1.0, 1.0]"
    
    # Test experience storage
    next_obs, reward, done, info = env.step(action)
    agent.remember(obs, action, reward, next_obs, done)
    print(f"✅ Experience storage works: buffer size = {len(agent.memory)}")
    
    # Test multiple steps to build buffer
    for step in range(50):
        action = agent.act(obs)
        next_obs, reward, done, info = env.step(action)
        agent.remember(obs, action, reward, next_obs, done)
        obs = next_obs
        
        if done:
            obs = env.reset()
    
    print(f"✅ Built replay buffer: {len(agent.memory)} experiences")
    
    # Test network updates (need sufficient buffer)
    if len(agent.memory) >= agent.batch_size:
        losses = agent.update_networks()
        print(f"✅ Network update works: {list(losses.keys())}")
        print(f"   Actor loss: {losses.get('actor_loss', 'N/A'):.4f}")
        print(f"   Critic losses: {losses.get('critic1_loss', 'N/A'):.4f}, {losses.get('critic2_loss', 'N/A'):.4f}")
        print(f"   Alpha: {losses.get('alpha_value', 'N/A'):.4f}")
    else:
        print(f"⚠️  Buffer too small for updates: {len(agent.memory)} < {agent.batch_size}")
    
    # Test model saving
    test_model_path = "test_sac_model.pth"
    agent.save_model(test_model_path)
    print(f"✅ Model saving works: {test_model_path}")
    
    # Test model loading
    new_agent = SACAgent(
        state_size=10,
        action_size=5,
        hidden_size=128
    )
    new_agent.load_model(test_model_path)
    print(f"✅ Model loading works")
    
    # Clean up
    Path(test_model_path).unlink()
    env.close()
    
    print("🎉 All SAC tests passed!")

def test_sac_short_training():
    """Test SAC with short training run"""
    print("\n🏃‍♂️ Testing SAC Short Training Run...")
    
    # Set seed
    random.seed(42)
    np.random.seed(42)
    
    # Initialize environment
    env = RestaurantDeliveryEnvironment(
        space_file=Path("grid_configs/my_first_restaurant.npz"), 
        enable_gui=False,
        max_episode_steps=100
    )
    
    # Initialize SAC agent with fast settings
    agent = SACAgent(
        state_size=10,
        action_size=5,
        lr=3e-3,      # Higher lr for faster learning
        gamma=0.95,   # Lower gamma for faster learning
        tau=0.01,     # Faster target updates
        alpha=0.1,    # Lower temperature
        buffer_size=2000,
        batch_size=64,
        hidden_size=128,
        auto_temp=False  # Fixed temperature for testing
    )
    
    # Quick training loop
    print("Training for 10 episodes...")
    
    episode_rewards = []
    for episode in range(10):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(100):  # Max 100 steps per episode
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.remember(obs, action, reward, next_obs, done)
            
            # Update networks if buffer is ready
            if len(agent.memory) >= agent.batch_size:
                agent.update_networks()
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if episode % 3 == 0:
            print(f"   Episode {episode}: Reward = {episode_reward:.2f}, Buffer = {len(agent.memory)}")
    
    print(f"✅ Training completed!")
    print(f"   Episodes: {len(episode_rewards)}")
    print(f"   Average reward: {np.mean(episode_rewards):.2f}")
    print(f"   Final buffer size: {len(agent.memory)}")
    print(f"   Total updates: {agent.update_count}")
    
    env.close()
    print("🎉 Short training test passed!")

if __name__ == "__main__":
    print("🚀 Testing Qi's SAC Implementation")
    print("=" * 50)
    
    try:
        test_sac_basic()
        test_sac_short_training()
        
        print("\n🎯 SUCCESS: SAC implementation is working correctly!")
        print("📋 Ready for full training with train_sac.py")
        print("🆚 Ready for comparison with Stefan's DQN")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1) 