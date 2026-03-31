def encrypt_vigenere(plaintext: str, keyword: str) -> str:

    ciphertext = ""

    for i, char in enumerate(plaintext):
        if char.isalpha():
            start = ord("A") if char.isupper() else ord("a")
            shift = ord(keyword[i % len(keyword)].lower()) - ord("a")
            ciphertext += chr((ord(char) - start + shift) % 26 + start)
        else:
            ciphertext += char
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    # PUT YOUR CODE HERE
    for i, char in enumerate(ciphertext):
        if char.isalpha():
            start = ord("A") if char.isupper() else ord("a")
            shift = ord(keyword[i % len(keyword)].lower()) - ord("a")
            # Дешифруем (вычитаем сдвиг)
            plaintext += chr((ord(char) - start - shift) % 26 + start)
        else:
            plaintext += char
    return plaintext
