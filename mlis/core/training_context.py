class TrainingContext():
    LOCAL_TRAINING = 'local_training'
    SERVER_TRAINING = 'server_training'
    GRID_SEARCH = 'grid_search'
    def __init__(self, run_seed, timer):
        self.type = self.__class__.LOCAL_TRAINING
        self.run_seed = run_seed
        self.timer = timer
        self.step = 0

    def increase_step(self):
        self.step += 1

    def get_reject_reason(self):
        if self.timer.pause_time > 0.0:
            return "Timer paused"
        return None
