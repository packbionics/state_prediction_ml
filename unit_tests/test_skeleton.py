from inspect import getmembers, ismethod


class Skeleton:
    def run_all_tests(self):
        """Calls all methods in self."""
        
        # Get all methods
        methods = getmembers(self, predicate=ismethod)
        
        # Filter out dunders and call each method
        [method() if not name.startswith('__') and not name.startswith('run') else None for name, method in methods]
