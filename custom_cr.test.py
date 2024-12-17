def test_large_number_of_obstacles():
    env = CustomCarRacing()
    env.create_obstacles = lambda: [(np.random.uniform(-1.0, 1.0), np.random.uniform(-1.0, 1.0)) for _ in range(1000)]
    
    start_time = time.time()
    env.reset()
    end_time = time.time()
    
    initialization_time = end_time - start_time
    assert initialization_time < 1.0, f"Initialization with 1000 obstacles took {initialization_time} seconds, which is too long"
    
    assert len(env.obstacles) == 1000, f"Expected 1000 obstacles, but got {len(env.obstacles)}"
    
    start_time = time.time()
    for _ in range(100):
        env.step(env.action_space.sample())
    end_time = time.time()
    
    step_time = (end_time - start_time) / 100
    assert step_time < 0.1, f"Average step time with 1000 obstacles is {step_time} seconds, which is too long"