import string
import re
from collections import Counter
import itertools

class CryptographicSolver:
    def __init__(self):
        # English letter frequencies (approximate)
        self.english_freq = {
            'E': 12.70, 'T': 9.06, 'A': 8.17, 'O': 7.51, 'I': 6.97,
            'N': 6.75, 'S': 6.33, 'H': 6.09, 'R': 5.99, 'D': 4.25,
            'L': 4.03, 'C': 2.78, 'U': 2.76, 'M': 2.41, 'W': 2.36,
            'F': 2.23, 'G': 2.02, 'Y': 1.97, 'P': 1.93, 'B': 1.29,
            'V': 0.98, 'K': 0.77, 'J': 0.15, 'X': 0.15, 'Q': 0.10,
            'Z': 0.07
        }
        
        # Common English words for pattern matching
        self.common_words = ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 
                             'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE',
                             'OUR', 'OUT', 'DAY', 'HIS', 'HOW', 'ITS']
        
        # Common bigrams and trigrams
        self.common_bigrams = ['TH', 'HE', 'IN', 'ER', 'AN', 'RE', 'ED', 
                               'ON', 'ES', 'ST', 'EN', 'AT', 'TO', 'NT']
        self.common_trigrams = ['THE', 'AND', 'THA', 'ENT', 'ING', 'ION',
                                'TIO', 'FOR', 'NDE', 'HAS', 'NCE', 'EDT']
    
    def solve_caesar_cipher(self, ciphertext):
        """Solve Caesar cipher using frequency analysis and brute force"""
        ciphertext = ciphertext.upper()
        best_score = -float('inf')
        best_shift = 0
        best_plaintext = ""
        
        print("Attempting Caesar cipher decryption...")
        
        for shift in range(26):
            plaintext = self.caesar_decrypt(ciphertext, shift)
            score = self.score_text(plaintext)
            
            if score > best_score:
                best_score = score
                best_shift = shift
                best_plaintext = plaintext
        
        return best_plaintext, best_shift
    
    def caesar_decrypt(self, text, shift):
        """Decrypt Caesar cipher with given shift"""
        result = []
        for char in text:
            if char in string.ascii_uppercase:
                shifted = (ord(char) - ord('A') - shift) % 26
                result.append(chr(shifted + ord('A')))
            else:
                result.append(char)
        return ''.join(result)
    
    def solve_substitution_cipher(self, ciphertext, max_iterations=10000):
        """Solve monoalphabetic substitution cipher using hill climbing"""
        ciphertext = ciphertext.upper()
        ciphertext_clean = re.sub(r'[^A-Z]', '', ciphertext)
        
        print("Attempting substitution cipher decryption...")
        
        # Initialize with frequency-based mapping
        current_key = self.initial_key_guess(ciphertext_clean)
        current_plaintext = self.apply_substitution(ciphertext, current_key)
        current_score = self.score_text(current_plaintext)
        
        best_key = current_key.copy()
        best_score = current_score
        best_plaintext = current_plaintext
        
        # Hill climbing with random restarts
        for iteration in range(max_iterations):
            # Make a small change to the key
            new_key = current_key.copy()
            
            # Use different mutation strategies
            if iteration % 100 == 0:  # Random restart
                new_key = self.random_key()
            else:
                # Swap two random letters
                i, j = random.sample(range(26), 2)
                letters = list(new_key.keys())
                new_key[letters[i]], new_key[letters[j]] = new_key[letters[j]], new_key[letters[i]]
            
            new_plaintext = self.apply_substitution(ciphertext, new_key)
            new_score = self.score_text(new_plaintext)
            
            # Accept if better or with small probability (simulated annealing)
            if new_score > current_score or random.random() < 0.01:
                current_key = new_key
                current_plaintext = new_plaintext
                current_score = new_score
                
                if current_score > best_score:
                    best_key = current_key.copy()
                    best_score = current_score
                    best_plaintext = current_plaintext
            
            if iteration % 1000 == 0:
                print(f"Iteration {iteration}, best score: {best_score:.2f}")
        
        return best_plaintext, best_key
    
    def solve_vigenere_cipher(self, ciphertext, max_key_length=20):
        """Solve Vigenere cipher using Kasiski examination and frequency analysis"""
        ciphertext = ciphertext.upper()
        ciphertext_clean = re.sub(r'[^A-Z]', '', ciphertext)
        
        print("Attempting Vigenere cipher decryption...")
        
        # Find probable key length using Index of Coincidence
        probable_lengths = self.find_key_length_ic(ciphertext_clean, max_key_length)
        
        best_plaintext = ""
        best_key = ""
        best_score = -float('inf')
        
        for key_length in probable_lengths[:3]:  # Try top 3 probable lengths
            print(f"Trying key length: {key_length}")
            
            # Split ciphertext into groups
            groups = [''] * key_length
            for i, char in enumerate(ciphertext_clean):
                groups[i % key_length] += char
            
            # Solve each group as Caesar cipher
            key = ""
            for group in groups:
                _, shift = self.solve_caesar_cipher(group)
                key += chr(shift + ord('A'))
            
            plaintext = self.vigenere_decrypt(ciphertext, key)
            score = self.score_text(plaintext)
            
            if score > best_score:
                best_score = score
                best_plaintext = plaintext
                best_key = key
        
        return best_plaintext, best_key
    
    def vigenere_decrypt(self, ciphertext, key):
        """Decrypt Vigenere cipher with given key"""
        result = []
        key_index = 0
        
        for char in ciphertext:
            if char in string.ascii_uppercase:
                shift = ord(key[key_index % len(key)]) - ord('A')
                decrypted = (ord(char) - ord('A') - shift) % 26
                result.append(chr(decrypted + ord('A')))
                key_index += 1
            else:
                result.append(char)
        
        return ''.join(result)
    
    def score_text(self, text):
        """Score text based on English language characteristics"""
        text_clean = re.sub(r'[^A-Z]', '', text.upper())
        if not text_clean:
            return -float('inf')
        
        score = 0
        
        # Frequency analysis
        freq = Counter(text_clean)
        total = len(text_clean)
        
        for letter, count in freq.items():
            expected = self.english_freq.get(letter, 0.01) / 100
            observed = count / total
            chi_squared = ((observed - expected) ** 2) / expected
            score -= chi_squared * 100
        
        # Bigram analysis
        bigrams = [text_clean[i:i+2] for i in range(len(text_clean)-1)]
        bigram_score = sum(1 for bg in bigrams if bg in self.common_bigrams)
        score += bigram_score * 5
        
        # Trigram analysis
        trigrams = [text_clean[i:i+3] for i in range(len(text_clean)-2)]
        trigram_score = sum(1 for tg in trigrams if tg in self.common_trigrams)
        score += trigram_score * 10
        
        # Word detection
        words = text.upper().split()
        word_score = sum(1 for word in words if word in self.common_words)
        score += word_score * 20
        
        return score
    
    def initial_key_guess(self, ciphertext):
        """Create initial substitution key based on frequency analysis"""
        # Count frequencies in ciphertext
        freq = Counter(ciphertext)
        
        # Sort letters by frequency
        cipher_letters = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
        english_letters = sorted(self.english_freq.keys(), 
                                key=lambda x: self.english_freq[x], reverse=True)
        
        # Create mapping
        key = {}
        for i, cipher_letter in enumerate(cipher_letters):
            if i < len(english_letters):
                key[cipher_letter] = english_letters[i]
        
        # Fill in missing letters
        all_letters = set(string.ascii_uppercase)
        unmapped_cipher = all_letters - set(key.keys())
        unmapped_plain = all_letters - set(key.values())
        
        for c, p in zip(unmapped_cipher, unmapped_plain):
            key[c] = p
        
        return key
    
    def random_key(self):
        """Generate random substitution key"""
        import random
        letters = list(string.ascii_uppercase)
        shuffled = letters.copy()
        random.shuffle(shuffled)
        return dict(zip(letters, shuffled))
    
    def apply_substitution(self, text, key):
        """Apply substitution cipher with given key"""
        result = []
        for char in text:
            if char in key:
                result.append(key[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def find_key_length_ic(self, ciphertext, max_length):
        """Find probable key length using Index of Coincidence"""
        ic_values = []
        
        for length in range(1, max_length + 1):
            # Split into groups
            groups = [''] * length
            for i, char in enumerate(ciphertext):
                groups[i % length] += char
            
            # Calculate average IC for groups
            total_ic = 0
            for group in groups:
                if len(group) > 1:
                    freq = Counter(group)
                    n = len(group)
                    ic = sum(freq[c] * (freq[c] - 1) for c in freq) / (n * (n - 1))
                    total_ic += ic
            
            avg_ic = total_ic / length
            ic_values.append((length, avg_ic))
        
        # Sort by IC value (higher is better)
        ic_values.sort(key=lambda x: x[1], reverse=True)
        return [length for length, _ in ic_values]
    
    def solve(self, ciphertext, cipher_type=None):
        """Main solving method"""
        if cipher_type:
            if cipher_type.lower() == 'caesar':
                return self.solve_caesar_cipher(ciphertext)
            elif cipher_type.lower() == 'substitution':
                return self.solve_substitution_cipher(ciphertext)
            elif cipher_type.lower() == 'vigenere':
                return self.solve_vigenere_cipher(ciphertext)
        else:
            # Try to detect cipher type
            print("Detecting cipher type...")
            
            # Try Caesar first (fastest)
            plaintext, shift = self.solve_caesar_cipher(ciphertext)
            score = self.score_text(plaintext)
            
            if score > -100:  # Reasonable threshold
                print(f"Detected as Caesar cipher with shift {shift}")
                return plaintext, f"Caesar (shift={shift})"
            
            # Try Vigenere
            plaintext, key = self.solve_vigenere_cipher(ciphertext)
            score = self.score_text(plaintext)
            
            if score > -100:
                print(f"Detected as Vigenere cipher with key '{key}'")
                return plaintext, f"Vigenere (key={key})"
            
            # Try substitution (slowest)
            plaintext, key = self.solve_substitution_cipher(ciphertext)
            return plaintext, "Substitution"


# Import at the top (was missing)
import random

# Test the solver
def test_crypto_solver():
    solver = CryptographicSolver()
    
    # Test Caesar cipher
    print("=" * 60)
    print("TEST 1: Caesar Cipher")
    print("=" * 60)
    ciphertext1 = "WKH TXLFN EURZQ IRA MXPSV RYHU WKH ODCB GRJ"
    plaintext1, shift1 = solver.solve_caesar_cipher(ciphertext1)
    print(f"Ciphertext: {ciphertext1}")
    print(f"Plaintext: {plaintext1}")
    print(f"Shift: {shift1}")
    
    # Test substitution cipher
    print("\n" + "=" * 60)
    print("TEST 2: Substitution Cipher")
    print("=" * 60)
    # "HELLO WORLD" with simple substitution
    ciphertext2 = "ITSSG VGKSR"
    plaintext2, key2 = solver.solve_substitution_cipher(ciphertext2, max_iterations=1000)
    print(f"Ciphertext: {ciphertext2}")
    print(f"Plaintext: {plaintext2}")
    
    # Test Vigenere cipher
    print("\n" + "=" * 60)
    print("TEST 3: Vigenere Cipher")
    print("=" * 60)
    # "ATTACKATDAWN" encrypted with key "LEMON"
    ciphertext3 = "LXFOPVEFRNHR"
    plaintext3, key3 = solver.solve_vigenere_cipher(ciphertext3)
    print(f"Ciphertext: {ciphertext3}")
    print(f"Plaintext: {plaintext3}")
    print(f"Key: {key3}")



if __name__ == "__main__":
    test_crypto_solver()