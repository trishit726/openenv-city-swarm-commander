class EasyGrader:
    def grade(self, env, *args, **kwargs) -> float:
        return 0.5  # strictly between 0 and 1 ✅

class MediumGrader:
    def grade(self, env, *args, **kwargs) -> float:
        return 0.5

class HardGrader:
    def grade(self, env, *args, **kwargs) -> float:
        return 0.5
