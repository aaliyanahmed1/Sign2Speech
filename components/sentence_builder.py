#!/usr/bin/env python3
"""
Sentence Builder Module
Converts sequences of detected signs into grammatically correct natural language sentences
"""

import re
from typing import List, Dict, Optional
from collections import Counter
import random

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("📥 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
except ImportError:
    print("⚠️ NLTK not available, using basic sentence building")
    NLTK_AVAILABLE = False


class SentenceBuilder:
    """Converts sign sequences to natural language sentences"""
    
    def __init__(self):
        """
        Initialize sentence builder with grammar rules and templates
        """
        # Grammar templates for different sentence types
        self.sentence_templates = {
            'greeting': [
                "{greeting}",
                "{greeting}, how are you?",
                "{greeting} there!"
            ],
            'request': [
                "Please {action}",
                "Can you {action}?",
                "I need {object}",
                "Could you help me {action}?"
            ],
            'statement': [
                "I {verb} {object}",
                "{subject} {verb} {object}",
                "I am {adjective}",
                "{subject} is {adjective}"
            ],
            'question': [
                "Do you {verb}?",
                "Are you {adjective}?",
                "What is {object}?",
                "Where is {object}?"
            ],
            'emotion': [
                "I feel {emotion}",
                "I am {emotion}",
                "That makes me {emotion}"
            ]
        }
        
        # Word categories for grammar rules
        self.word_categories = {
            'pronouns': ['I', 'you', 'he', 'she', 'we', 'they', 'it'],
            'verbs': ['love', 'like', 'want', 'need', 'have', 'give', 'eat', 'drink', 'go', 'come', 'help', 'work', 'play', 'learn', 'see', 'hear'],
            'nouns': ['food', 'water', 'home', 'work', 'school', 'friend', 'family', 'book', 'car', 'phone'],
            'adjectives': ['good', 'bad', 'happy', 'sad', 'big', 'small', 'hot', 'cold', 'fast', 'slow'],
            'greetings': ['hello', 'hi', 'goodbye', 'bye'],
            'emotions': ['happy', 'sad', 'angry', 'excited', 'tired', 'confused'],
            'actions': ['help', 'stop', 'go', 'come', 'eat', 'drink', 'sleep', 'work', 'play'],
            'courtesy': ['please', 'thank_you', 'sorry', 'excuse_me'],
            'responses': ['yes', 'no', 'maybe', 'okay']
        }
        
        # Common sign language phrase patterns
        self.phrase_patterns = {
            ('I', 'love', 'you'): "I love you",
            ('thank', 'you'): "Thank you",
            ('thank_you',): "Thank you",
            ('please', 'help'): "Please help me",
            ('how', 'are', 'you'): "How are you?",
            ('nice', 'to', 'meet', 'you'): "Nice to meet you",
            ('good', 'morning'): "Good morning",
            ('good', 'night'): "Good night",
            ('see', 'you', 'later'): "See you later",
            ('I', 'am', 'fine'): "I am fine",
            ('what', 'is', 'your', 'name'): "What is your name?",
            ('my', 'name', 'is'): "My name is",
            ('where', 'are', 'you', 'from'): "Where are you from?"
        }
        
        # Word normalization mapping
        self.word_normalization = {
            'thank_you': 'thank you',
            'good_morning': 'good morning',
            'good_night': 'good night',
            'excuse_me': 'excuse me',
            'nice_to_meet_you': 'nice to meet you',
            'see_you_later': 'see you later',
            'how_are_you': 'how are you',
            'i_love_you': 'I love you'
        }
        
        print("📝 Sentence Builder initialized")
    
    def build_sentence(self, sign_sequence: List[str]) -> str:
        """
        Build a complete sentence from a sequence of signs
        
        Args:
            sign_sequence: List of detected sign class names
            
        Returns:
            Grammatically correct sentence string
        """
        if not sign_sequence:
            return ""
        
        # Clean and normalize the sequence
        cleaned_sequence = self._clean_sequence(sign_sequence)
        
        if not cleaned_sequence:
            return ""
        
        # Try to match known phrase patterns first
        sentence = self._match_phrase_patterns(cleaned_sequence)
        if sentence:
            return sentence
        
        # Try to build sentence using grammar rules
        sentence = self._build_with_grammar_rules(cleaned_sequence)
        if sentence:
            return sentence
        
        # Fallback: simple concatenation with basic grammar
        return self._simple_sentence_building(cleaned_sequence)
    
    def build_partial_sentence(self, sign_sequence: List[str]) -> str:
        """
        Build a partial sentence for real-time display
        
        Args:
            sign_sequence: Current sequence of signs
            
        Returns:
            Partial sentence or word sequence
        """
        if not sign_sequence:
            return "[Listening...]"
        
        cleaned_sequence = self._clean_sequence(sign_sequence)
        
        if len(cleaned_sequence) == 1:
            return self._normalize_word(cleaned_sequence[0])
        
        # Try to match partial patterns
        partial = self._match_partial_patterns(cleaned_sequence)
        if partial:
            return partial + "..."
        
        # Simple concatenation for partial display
        normalized_words = [self._normalize_word(word) for word in cleaned_sequence]
        return " ".join(normalized_words) + "..."
    
    def _clean_sequence(self, sequence: List[str]) -> List[str]:
        """
        Clean and filter the sign sequence
        """
        if not sequence:
            return []
        
        # Remove duplicates while preserving order
        seen = set()
        cleaned = []
        for sign in sequence:
            if sign and sign.lower() not in seen:
                cleaned.append(sign.lower())
                seen.add(sign.lower())
        
        # Remove very short sequences that might be noise
        if len(cleaned) == 1 and len(cleaned[0]) < 2:
            return []
        
        return cleaned
    
    def _match_phrase_patterns(self, sequence: List[str]) -> Optional[str]:
        """
        Match against known phrase patterns
        """
        sequence_tuple = tuple(sequence)
        
        # Exact match
        if sequence_tuple in self.phrase_patterns:
            return self.phrase_patterns[sequence_tuple]
        
        # Partial match (sequence contains a known pattern)
        for pattern, phrase in self.phrase_patterns.items():
            if self._sequence_contains_pattern(sequence, pattern):
                return phrase
        
        return None
    
    def _match_partial_patterns(self, sequence: List[str]) -> Optional[str]:
        """
        Match partial patterns for real-time display
        """
        sequence_tuple = tuple(sequence)
        
        # Check if current sequence is the beginning of any known pattern
        for pattern, phrase in self.phrase_patterns.items():
            if len(sequence_tuple) < len(pattern):
                if pattern[:len(sequence_tuple)] == sequence_tuple:
                    return phrase
        
        return None
    
    def _sequence_contains_pattern(self, sequence: List[str], pattern: tuple) -> bool:
        """
        Check if sequence contains the pattern
        """
        if len(pattern) > len(sequence):
            return False
        
        for i in range(len(sequence) - len(pattern) + 1):
            if tuple(sequence[i:i+len(pattern)]) == pattern:
                return True
        
        return False
    
    def _build_with_grammar_rules(self, sequence: List[str]) -> Optional[str]:
        """
        Build sentence using grammar rules and templates
        """
        # Categorize words
        categorized = self._categorize_words(sequence)
        
        # Determine sentence type
        sentence_type = self._determine_sentence_type(categorized)
        
        if sentence_type not in self.sentence_templates:
            return None
        
        # Select appropriate template
        templates = self.sentence_templates[sentence_type]
        template = random.choice(templates)
        
        # Fill template with words from sequence
        try:
            filled_sentence = self._fill_template(template, categorized)
            return filled_sentence
        except:
            return None
    
    def _categorize_words(self, sequence: List[str]) -> Dict[str, List[str]]:
        """
        Categorize words in the sequence
        """
        categorized = {category: [] for category in self.word_categories}
        
        for word in sequence:
            for category, words in self.word_categories.items():
                if word in words:
                    categorized[category].append(word)
        
        return categorized
    
    def _determine_sentence_type(self, categorized: Dict[str, List[str]]) -> str:
        """
        Determine the type of sentence based on categorized words
        """
        if categorized['greetings']:
            return 'greeting'
        elif categorized['emotions']:
            return 'emotion'
        elif 'please' in categorized['courtesy'] or categorized['actions']:
            return 'request'
        elif any(word in ['what', 'where', 'how', 'when', 'why'] for word in categorized.get('pronouns', [])):
            return 'question'
        else:
            return 'statement'
    
    def _fill_template(self, template: str, categorized: Dict[str, List[str]]) -> str:
        """
        Fill template with appropriate words
        """
        filled = template
        
        # Replace placeholders
        replacements = {
            '{greeting}': categorized['greetings'][0] if categorized['greetings'] else 'hello',
            '{subject}': categorized['pronouns'][0] if categorized['pronouns'] else 'I',
            '{verb}': categorized['verbs'][0] if categorized['verbs'] else 'am',
            '{object}': categorized['nouns'][0] if categorized['nouns'] else 'something',
            '{adjective}': categorized['adjectives'][0] if categorized['adjectives'] else 'good',
            '{emotion}': categorized['emotions'][0] if categorized['emotions'] else 'happy',
            '{action}': categorized['actions'][0] if categorized['actions'] else 'help'
        }
        
        for placeholder, replacement in replacements.items():
            if placeholder in filled:
                filled = filled.replace(placeholder, replacement)
        
        return filled.capitalize()
    
    def _simple_sentence_building(self, sequence: List[str]) -> str:
        """
        Fallback: simple sentence building with basic grammar
        """
        if not sequence:
            return ""
        
        # Normalize words
        normalized = [self._normalize_word(word) for word in sequence]
        
        # Basic sentence construction
        if len(normalized) == 1:
            word = normalized[0]
            if word in self.word_categories['greetings']:
                return word.capitalize() + "!"
            elif word in self.word_categories['courtesy']:
                return word.capitalize() + "."
            else:
                return word.capitalize() + "."
        
        # Multiple words - try to form a basic sentence
        sentence = " ".join(normalized)
        
        # Add appropriate punctuation
        if any(word in sentence.lower() for word in ['what', 'where', 'how', 'when', 'why', 'do', 'are']):
            sentence += "?"
        else:
            sentence += "."
        
        return sentence.capitalize()
    
    def _normalize_word(self, word: str) -> str:
        """
        Normalize a word (handle underscores, etc.)
        """
        if word in self.word_normalization:
            return self.word_normalization[word]
        
        # Replace underscores with spaces
        normalized = word.replace('_', ' ')
        
        return normalized
    
    def add_custom_pattern(self, pattern: tuple, sentence: str):
        """
        Add a custom phrase pattern
        
        Args:
            pattern: Tuple of sign sequence
            sentence: Corresponding sentence
        """
        self.phrase_patterns[pattern] = sentence
        print(f"➕ Added custom pattern: {pattern} -> '{sentence}'")
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the sentence builder
        """
        return {
            'total_patterns': len(self.phrase_patterns),
            'word_categories': {cat: len(words) for cat, words in self.word_categories.items()},
            'sentence_templates': {stype: len(templates) for stype, templates in self.sentence_templates.items()}
        }


def test_sentence_builder():
    """Test function for sentence builder"""
    print("🧪 Testing Sentence Builder...")
    
    builder = SentenceBuilder()
    
    # Test cases
    test_cases = [
        ['hello'],
        ['I', 'love', 'you'],
        ['thank_you'],
        ['please', 'help'],
        ['good', 'morning'],
        ['I', 'am', 'happy'],
        ['where', 'are', 'you'],
        ['nice', 'to', 'meet', 'you'],
        ['I', 'want', 'food'],
        ['you', 'are', 'good']
    ]
    
    print("\n📝 Testing sentence building:")
    for i, sequence in enumerate(test_cases, 1):
        sentence = builder.build_sentence(sequence)
        partial = builder.build_partial_sentence(sequence)
        print(f"  {i}. {sequence} -> '{sentence}' (partial: '{partial}')")
    
    # Test statistics
    stats = builder.get_statistics()
    print(f"\n📊 Builder Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    test_sentence_builder()