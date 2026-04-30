import re
from collections import defaultdict

class CompositionAnalyzer:
    """Analyze failures by compositional phenomena."""
    
    @staticmethod
    def detect_phenomena(input_text):
        """
        Detect compositional phenomena in input.
        
        Phenomena to detect:
        - NESTING: Repeated actions ("turn left turn left")
        - MODIFIERS: Actions with count modifiers ("turn left twice")
        - CONJUNCTIONS: Multiple actions connected ("turn left and walk")
        - LENGTH: Long sequences vs short
        """
        phenomena = []

        if any(mod in input_text for mod in ['twice', 'thrice', 'four times']):
            phenomena.append('MODIFIER')
        if len(input_text.split()) > 4:
            phenomena.append('NESTING')
        if ' and ' in input_text:
            phenomena.append('CONJUNCTION')
        return phenomena

    
    @staticmethod
    def analyze_failures_by_phenomena(errors):
        """
        Group failures by phenomena.
        
        Args:
            errors: List of error dicts from ErrorAnalyzer
        
        Returns:
            Dictionary like:
            {
                'NESTING': {'count': 5, 'rate': 0.1, 'examples': [...]},
                'MODIFIERS': {'count': 2, 'rate': 0.05, 'examples': [...]},
                ...
            }
        """
        phenomenon_dict = defaultdict(lambda:{
            'count': 0,
            'examples': [],
            'rate': 0.0
        })

        for error in errors:
            input_text = error['input']
            phenomena = CompositionAnalyzer.detect_phenomena(input_text)

            for phenomenon in phenomena:
                phenomenon_dict[phenomenon]['count'] += 1
                if len(phenomenon_dict[phenomenon]['examples']) < 3:
                    phenomenon_dict[phenomenon]['examples'].append(error)
        
        total_errors = len(errors)
        for phenomenon in phenomenon_dict:
            phenomenon_dict[phenomenon]['rate'] = (
                phenomenon_dict[phenomenon]['count'] / total_errors 
                if total_errors > 0 else 0
            )
        
        return dict(phenomenon_dict)