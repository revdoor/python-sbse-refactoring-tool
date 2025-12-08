def process(user, min_age):
    if user.age >= min_age and user.is_active:
        return "allowed"
    return "denied"

class Validator:
    def __init__(self):
        self.min_age = 18
    
    def check(self, user):
        if user.age >= self.min_age and user.is_active:
            return True
        return False
    
class Calculator:
    def compute(self, x, y):
        if x > 0 and y > 0:
            return x + y
        return 0
