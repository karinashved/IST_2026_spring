import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    ciphertext = ""
    for char in plaintext:
        if char.isalpha():
            start = ord("A") if char.isupper() else ord("a")
            ciphertext += chr((ord(char) - start + shift) % 26 + start)
        else:
            ciphertext += char
    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    plaintext = encrypt_caesar(ciphertext, -shift)
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    # PUT YOUR CODE HERE
    best_shift = 0
    max_matches = 0
    for shift in range(26):
        decrypted = decrypt_caesar(ciphertext, shift)
        matches = sum(1 for word in decrypted.split() if word.lower() in dictionary)
        if matches > max_matches:
            max_matches = matches
            best_shift = shift
    return best_shift
