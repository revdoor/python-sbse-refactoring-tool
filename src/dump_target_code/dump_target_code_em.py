# # ============================================================
# # ============================================================
# def process_data(items):
#     total = 0
#     for item in items:
#         total += item
#     print(f"Processing {len(items)} items")
#     print(f"Total: {total}")
#     return total


# ============================================================
# ============================================================
class DataProcessor:
    def __init__(self):
        self.log_enabled = True
        self.prefix = "[LOG]"
    
    def process(self, data):
        result = []
        for item in data:
            result.append(item * 2)
        if self.log_enabled:
            print(f"{self.prefix} Processed {len(data)} items")
            print(f"{self.prefix} Result size: {len(result)}")
        return result


# # ============================================================
# # ============================================================
# class Calculator:
#     def __init__(self):
#         self.history = []
    
#     def compute(self, a, b):
#         x = a + b
#         y = a * b
#         print(f"Sum: {x}")
#         print(f"Product: {y}")
#         self.history.append((a, b, x, y))
#         return x, y
