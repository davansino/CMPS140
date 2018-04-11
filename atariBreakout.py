# Import the gym module
import gym
import time

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
#env = gym.make('MountainCar-v0')
#env = gym.make('MsPacman-v0')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

done = False
try: 
  while not done:
    # Perform a random action, returns the new frame, reward and whether the game is over
    observation, reward, done, info = env.step(env.action_space.sample())
    # Render
    env.render()

    #action = env.action_space.sample()
    #observation, reward, done, info = env.step(action)

    #print(observation, reward, done, info)
    print(env.action_space)
    #> Discrete(2)
    print(env.observation_space)
    #> Box(4,)

    #sleep
    time.sleep(0.05)
except KeyboardInterrupt:
    pass
