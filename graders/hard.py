class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        """
        Grades the 'hard' task. 
        Note: The score is already clamped between 0.01 and 0.99 in the environment.
        """
        return float(env.current_mission_score)
